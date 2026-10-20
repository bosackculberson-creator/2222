#!/usr/bin/env python3
"""Animate the supplied pixel background without rescaling its canvas.

Dependencies: Pillow, numpy, scipy. No generative repainting or frame interpolation.
Run: python render_animation.py source.png destination-folder
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d

FRAME_COUNT = 30
DURATION_MS = 100
LOOP_SECONDS = FRAME_COUNT * DURATION_MS / 1000
REFERENCE_SIZE = (1983, 793)

# Explicit water channels; architectural columns and cliff faces are excluded.
WATER_CHANNELS = [
    {"name": "main", "crest": 371, "end": 540,
     "polygon": [(1290,371),(1307,370),(1311,391),(1314,486),
                 (1317,524),(1309,543),(1294,535),(1285,492),(1288,397)]},
    {"name": "left", "crest": 383, "end": 495,
     "polygon": [(576,383),(588,385),(594,408),(597,477),
                 (590,498),(579,486),(576,411)]},
    {"name": "lower_middle", "crest": 512, "end": 626,
     "polygon": [(1055,512),(1067,513),(1073,540),(1077,604),
                 (1072,627),(1060,618),(1056,564)]},
    {"name": "right_upper", "crest": 420, "end": 535,
     "polygon": [(1576,419),(1585,419),(1589,440),(1592,518),
                 (1587,537),(1579,530),(1577,462)]},
    {"name": "right_lower", "crest": 470, "end": 549,
     "polygon": [(1590,470),(1598,469),(1603,490),(1605,535),
                 (1599,550),(1591,541)]},
]


def smoothstep(a, b, x):
    z = np.clip((x - a) / (b - a), 0, 1)
    return z * z * (3 - 2 * z)


def polygon_mask(size, points):
    w, h = size
    sx, sy = w / REFERENCE_SIZE[0], h / REFERENCE_SIZE[1]
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon([(round(x*sx), round(y*sy)) for x,y in points], fill=255)
    return np.asarray(canvas, dtype=np.float32) / 255


def encode_gif(frames, destination, colors=192):
    """Use a global palette plus transparent deltas to avoid palette shimmer."""
    w,h=frames[0].size
    samples=Image.new("RGB",(w,h*3))
    for n,i in enumerate([0,10,20]): samples.paste(frames[i],(0,n*h))
    palette=samples.quantize(colors=colors,method=Image.Quantize.MEDIANCUT)
    palette_data=palette.getpalette()
    delta_frames=[]
    previous=None
    expected=[]
    for frame in frames:
        q=frame.quantize(palette=palette,dither=Image.Dither.NONE)
        pixels=np.asarray(q).copy()
        assert pixels.max()<255
        expected.append(np.asarray(q.convert("RGB")))
        delta=pixels.copy()
        if previous is not None:
            delta[pixels==previous]=255
        item=Image.fromarray(delta,"P")
        item.putpalette(palette_data)
        item.info["transparency"]=255
        delta_frames.append(item)
        previous=pixels
    delta_frames[0].save(destination,save_all=True,append_images=delta_frames[1:],
        duration=DURATION_MS,loop=0,disposal=1,optimize=False,transparency=255)
    gif=Image.open(destination)
    assert gif.n_frames==len(frames)
    for index,reference in enumerate(expected):
        gif.seek(index)
        assert np.array_equal(np.asarray(gif.convert("RGB")),reference),index


class BackgroundAnimator:
    def __init__(self, source):
        self.source = Path(source)
        self.image = Image.open(source).convert("RGB")
        self.base = np.asarray(self.image, dtype=np.float32)
        self.h, self.w = self.base.shape[:2]
        self.sx = self.w / REFERENCE_SIZE[0]
        self.sy = self.h / REFERENCE_SIZE[1]
        self.y, self.x = np.mgrid[:self.h,:self.w].astype(np.float32)
        self.rng = np.random.default_rng(20261020)
        self.channels = []
        self.water_mask = np.zeros((self.h,self.w), np.float32)
        r,g,b = np.moveaxis(self.base, -1, 0)
        for spec in WATER_CHANNELS:
            region = polygon_mask(self.image.size, spec["polygon"])
            # The existing cyan water supplies the matte, not the rectangular ROI.
            cyan = smoothstep(14,47,g-r) * smoothstep(12,45,b-g)
            luminosity = smoothstep(62,125,b)
            water = region * cyan * luminosity
            water = np.minimum(region, gaussian_filter(water, (0.5*self.sy,0.65*self.sx)))
            crest, end = spec["crest"]*self.sy, spec["end"]*self.sy
            depth = np.maximum(0,self.y-crest)
            v0 = 24*self.sy
            gravity = 360*self.sy
            fall_time = (np.sqrt(v0*v0+2*gravity*depth)-v0)/gravity
            fade = smoothstep(1*self.sy,8*self.sy,depth)*(1-smoothstep(end-20*self.sy,end+3*self.sy,self.y))
            water *= fade
            self.water_mask = np.maximum(self.water_mask, water)
            # Periodic Lagrangian texture, advected in fall-time coordinates.
            # q(y)-t = constant -> dy/dt = sqrt(v0^2+2*g*y), always downward.
            table = self.rng.normal(size=(360,40)).astype(np.float32)
            table = gaussian_filter(table,(15,0.65),mode=("wrap","reflect"))
            table /= max(float(np.std(table)),1e-6)
            self.channels.append(dict(spec=spec, mask=water, fall_time=fall_time,
                                      table=table, crest=crest, end=end))

        # Select blue low-contrast vapor. Dark mountains and warm architecture stay pinned.
        blue = smoothstep(22,48,b-g)*smoothstep(12,30,g-r)
        light = smoothstep(65,112,b)*(1-smoothstep(155,205,b))
        low = gaussian_filter(self.base,(3*self.sy,5*self.sx,0))
        detail = np.max(np.abs(self.base-low),axis=2)
        quiet = 1-smoothstep(9,24,gaussian_filter(detail,1.2*self.sy))
        self.fog_mask = blue*light*quiet*smoothstep(145*self.sy,340*self.sy,self.y)
        self.fog_mask *= 1-self.water_mask
        self.fog_mask = np.clip(self.fog_mask,0,1)
        # Sky cloud matte avoids the far mountain skyline and all roofs.
        sky_zone = polygon_mask(self.image.size,[(0,0),(1983,0),(1983,135),
            (1730,137),(1660,90),(1595,62),(1555,0),(1490,0),
            (1450,80),(1360,110),(1120,144),(980,129),(850,120),
            (695,95),(600,122),(425,142),(275,127),(120,35),(0,40)])
        self.sky_mask = sky_zone*smoothstep(57,112,b)*blue
        protected = np.zeros((self.h,self.w),np.float32)
        for points in [
            [(593,110),(834,110),(892,362),(609,367)],
            [(1115,250),(1350,250),(1403,366),(1145,381)],
            [(993,342),(1257,342),(1257,409),(993,409)],
            [(1289,0),(1731,0),(1731,324),(1368,332),(1289,250)],
            [(410,277),(493,277),(493,370),(410,370)],
            [(886,267),(1018,267),(1031,368),(886,365)],
            [(1592,267),(1670,267),(1670,335),(1592,335)],
        ]:
            protected = np.maximum(protected,polygon_mask(self.image.size,points))
        self.fog_mask *= 1-protected
        self.sky_mask *= 1-protected
        self.x_int,self.y_int = np.meshgrid(np.arange(self.w),np.arange(self.h))
        # Two counterphased flow layers fade before either texture is reset.
        # Both layers are transported to the right; no backwards frame playback.
        self.fog_texture = self.rng.normal(size=(max(8,self.h//3),64)).astype(np.float32)
        self.fog_texture = gaussian_filter(self.fog_texture,(4,6),mode=("reflect","wrap"))
        self.fog_texture /= max(float(np.std(self.fog_texture)),1e-6)
        self.static_mask = np.maximum(self.fog_mask,self.sky_mask)<0.003
        self.static_mask &= self.water_mask<0.003

    def cloud_drift(self, time):
        phase = time / LOOP_SECONDS
        out=self.base.copy()
        for mask,displacement in [(self.fog_mask,42),(self.sky_mask,25)]:
            for offset in (0,0.5):
                p=(phase+offset)%1
                weight=math.sin(math.pi*p)**2
                dx=int(round((p-0.5)*displacement*self.sx))
                source_x=np.clip(self.x_int-dx,0,self.w-1)
                safe_mask=mask*mask[self.y_int,source_x]
                delta=self.base[self.y_int,source_x]-self.base
                out+=delta*(safe_mask*weight*0.85)[...,None]
        # Advected, tileable wisps give slow transport even in featureless vapor.
        # Sampling stays on the source pixel grid, preserving pixel edges.
        field=self.fog_texture
        fy=np.clip((self.y_int//3),0,field.shape[0]-1)
        fx=(self.x_int//2-int(round(phase*field.shape[1])))%field.shape[1]
        noise=field[fy,fx]
        delta=noise*(self.fog_mask*3+self.sky_mask*1.4)
        out+=delta[...,None]*np.array([0.45,0.80,1.15],np.float32)
        return out

    def frame(self, index):
        time = (index % FRAME_COUNT)*DURATION_MS/1000
        out = self.cloud_drift(time)
        for channel in self.channels:
            tab = channel["table"]
            rowf = ((channel["fall_time"]-time)/LOOP_SECONDS*len(tab))%len(tab)
            i0 = np.floor(rowf).astype(np.int32)
            i1 = (i0+1)%len(tab)
            frac = rowf-i0
            columns = (np.floor(self.x/max(1,round(self.sx))).astype(np.int32))%tab.shape[1]
            tex = tab[i0,columns]*(1-frac)+tab[i1,columns]*frac
            tex = np.clip(tex,-2,2)
            # Long downward streaks accelerate and stretch with falling distance.
            addition = tex[...,None]*np.array([9,17,23],np.float32)
            out += addition*channel["mask"][...,None]
        out = np.clip(np.rint(out),0,255).astype(np.uint8)
        # Guarantee immutable pixels outside the selected atmospheric/water mattes.
        out[self.static_mask] = self.base.astype(np.uint8)[self.static_mask]
        return Image.fromarray(out,"RGB")

    def render(self, destination):
        destination=Path(destination)
        frames_dir=destination/"frames"
        frames_dir.mkdir(parents=True,exist_ok=True)
        frames=[]
        for index in range(FRAME_COUNT):
            frame=self.frame(index)
            frame.save(frames_dir/f"frame_{index+1:03d}.png",compress_level=7)
            frames.append(frame)
            print(f"Rendered {index+1}/{FRAME_COUNT}",flush=True)
        encode_gif(frames,destination/"background.gif")
        shutil.copyfile(self.source,destination/"source.png")
        manifest={"width":self.w,"height":self.h,"frame_count":FRAME_COUNT,
            "fps":1000/DURATION_MS,"duration_ms":FRAME_COUNT*DURATION_MS,
            "frame_duration_ms":DURATION_MS,"loop":True,
            "sequence":"frames/frame_%03d.png","first_frame":1,
            "last_frame":FRAME_COUNT,"source_sha256":hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "motion":{"water":"downward, v(y)=sqrt(v0²+2gy); fixed crests; periodic inflow",
                      "clouds":"periodic horizontal atmospheric transport; pinned architecture"}}
        (destination/"animation.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
        # Meaningful file, timing, closure and scenery-lock checks.
        hashes=[hashlib.sha256(np.asarray(f).tobytes()).hexdigest() for f in frames]
        assert len(set(hashes))==FRAME_COUNT,"Frames must all be distinct"
        assert np.array_equal(np.asarray(self.frame(0)),np.asarray(self.frame(FRAME_COUNT)))
        diffs=[float(np.mean(np.abs(np.asarray(frames[(i+1)%FRAME_COUNT],dtype=float)-
                                    np.asarray(frames[i],dtype=float)))) for i in range(FRAME_COUNT)]
        gif=Image.open(destination/"background.gif")
        assert gif.n_frames==FRAME_COUNT
        assert gif.info.get("loop")==0
        durations=[]
        for i in range(FRAME_COUNT):
            gif.seek(i);durations.append(gif.info["duration"])
        assert durations==[DURATION_MS]*FRAME_COUNT
        report={"passed":True,"png_frames":len(hashes),"unique_frames":len(set(hashes)),
            "gif_frames":gif.n_frames,"gif_durations_ms":durations,
            "exact_periodic_closure":True,"mean_adjacent_difference":float(np.mean(diffs)),
            "wrap_difference":diffs[-1],"max_adjacent_difference":max(diffs),
            "fixed_pixel_fraction":float(np.mean(self.static_mask))}
        (destination/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    BackgroundAnimator(args.source).render(args.destination)
