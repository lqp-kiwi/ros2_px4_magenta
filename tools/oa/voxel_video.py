#!/usr/bin/env python3
"""
Video of the 3DVFH+ point memory of a flight as 2D voxels seen from above: voxels only.

No text, axes, legend, drone or path: every frame is the planner's memory after the last depth
image, each 0.2 m voxel a filled square (north up, east right), coloured by height (the highest
voxel of a column on top), on black. The frame is fixed and holds every voxel of the flight.

Source: vfh.npz of a flight log folder, vox_* arrays (logged since 2026-11-20 by the
obstacle_avoidance_static_* nodes, also written by tools/oa/depth_filter_check.py).

Usage:
    python3 tools/oa/voxel_video.py <folder> [--speed X] [--out file.mp4] [--frames t1 t2 ...]
Output: <folder>/voxel_video.mp4 (H.264, 1280x720, 10 frames per second, --speed 2 plays the flight
twice as fast), or PNGs with --frames. Needs PIL and gst-launch-1.0 with x264enc and mp4mux
(the PC, not the Orin).

Last Modified: 2026-11-20
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

FPS = 10
W, H = 1280, 720            # video size (px)
MARGIN = 10                 # free border (px)


def turbo(x):
    """Turbo colour map (polynomial fit by Google), x in 0..1 -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)[..., None]
    c = np.array([[0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396,
                   59.28637943],
                  [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
                  [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912,
                   27.34824973]])
    rgb = sum(c[:, k] * x ** k for k in range(6))
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def pack(idx):
    """Voxel indices (N, 3) -> one int64 each, for set operations (as vfh3d.voxel_keys)."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3) + (1 << 20)
    return (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]


def unpack(keys):
    mask = (1 << 21) - 1
    return np.stack([(keys >> 42) & mask, (keys >> 21) & mask, keys & mask], axis=1) - (1 << 20)


def encode(frame_dir, out):
    pipeline = (f"multifilesrc location={frame_dir}/f%05d.png index=0 "
                f"caps=image/png,framerate={FPS}/1 ! pngdec ! videoconvert ! "
                "video/x-raw,format=I420 ! x264enc speed-preset=medium bitrate=3000 ! "
                f"video/x-h264,profile=main ! mp4mux ! filesink location={out}")
    subprocess.run(["gst-launch-1.0", "-q"] + pipeline.split(), check=True,
                   stdout=subprocess.DEVNULL)      # newer gst prints its progress even with -q


class Memory:
    """The point memory at any time, rebuilt from the logged changes (forward in time only)."""

    def __init__(self, npz):
        self.t = npz["vox_t"]
        self.n = -1
        self.keys = np.empty(0, dtype=np.int64)
        self.ev = {}
        for k in ("add", "del"):
            self.ev[k] = (npz[f"vox_{k}"], np.concatenate([[0], np.cumsum(npz[f"vox_{k}_n"])]))

    def part(self, k, i):
        data, off = self.ev[k]
        return data[off[i]:off[i + 1]]

    def at(self, t):
        """Voxel indices (N, 3) in the memory at time t."""
        k = max(int(np.searchsorted(self.t, t, side="right")) - 1, 0)
        while self.n < k:
            self.n += 1
            self.keys = np.setdiff1d(self.keys, pack(self.part("del", self.n)), assume_unique=True)
            self.keys = np.union1d(self.keys, pack(self.part("add", self.n)))
        return unpack(self.keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed, default 1 (real time)")
    ap.add_argument("--out")
    ap.add_argument("--frames", type=float, nargs="*",
                    help="only write PNG frames at these times (s)")
    args = ap.parse_args()

    folder = os.path.abspath(os.path.expanduser(args.folder))
    npz = np.load(os.path.join(folder, "vfh.npz"))
    if "vox_t" not in npz.files:
        sys.exit("vfh.npz has no point-memory log (vox_*, logged since 2026-11-20)")
    mem = Memory(npz)
    if len(mem.t) == 0 or len(npz["vox_add"]) == 0:
        sys.exit("vfh.npz has no voxel in its point-memory log")

    # Fixed frame around every voxel of the flight. Voxel index: north, east, down.
    allv = npz["vox_add"].astype(np.int64)
    n0, n1 = allv[:, 0].min(), allv[:, 0].max()
    e0, e1 = allv[:, 1].min(), allv[:, 1].max()
    up = -allv[:, 2]
    lo, hi = np.percentile(up, 2), np.percentile(up, 98)
    rows, cols = n1 - n0 + 1, e1 - e0 + 1
    group = max(1, math.ceil(rows / (H - 2 * MARGIN)), math.ceil(cols / (W - 2 * MARGIN)))
    rows, cols = math.ceil(rows / group), math.ceil(cols / group)
    px = max(1, min((H - 2 * MARGIN) // rows, (W - 2 * MARGIN) // cols))   # pixels per cell
    y0, x0 = (H - rows * px) // 2, (W - cols * px) // 2

    def frame(t):
        v = mem.at(t).astype(np.int64)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        if len(v) == 0:
            return img
        r = (n1 - v[:, 0]) // group             # north up
        c = (v[:, 1] - e0) // group             # east right
        top = np.full((rows, cols), np.iinfo(np.int64).min)
        np.maximum.at(top, (r, c), -v[:, 2])    # highest voxel of each column
        seen = top > np.iinfo(np.int64).min
        cell = np.zeros((rows, cols, 3), dtype=np.uint8)
        cell[seen] = turbo((top[seen] - lo) / max(hi - lo, 1))
        img[y0:y0 + rows * px, x0:x0 + cols * px] = np.repeat(np.repeat(cell, px, 0), px, 1)
        return img

    t = mem.t
    if args.frames:
        outdir = os.path.dirname(os.path.abspath(args.out)) if args.out else folder
        for ft in sorted(args.frames):
            p = os.path.join(outdir, f"voxel_{ft:05.1f}.png")
            Image.fromarray(frame(ft)).save(p)
            print(p)
        return

    out = args.out or os.path.join(folder, "voxel_video.mp4")
    out = os.path.abspath(os.path.expanduser(out))
    times = np.arange(t[0], t[-1] + 1e-6, args.speed / FPS)
    tmp = tempfile.mkdtemp(prefix="voxel_video_")
    try:
        for i, ft in enumerate(times):
            Image.fromarray(frame(ft)).save(os.path.join(tmp, f"f{i:05d}.png"), compress_level=1)
        encode(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{out}: {len(times)} frames, {len(times) / FPS:.0f} s (x{args.speed:g})")


if __name__ == "__main__":
    main()
