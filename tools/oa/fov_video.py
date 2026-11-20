#!/usr/bin/env python3
"""
Video of the depth camera field of view as a grid of 6 x 6 deg cells, each showing a distance (m).

Source: vfh.npz of a flight log folder (FlightLogger of the obstacle_avoidance_static_* nodes), one
frame per 3DVFH+ step (10 Hz), played at real time. Labels are in Vietnamese.

Modes (what the number in a cell is):
    frame   (default) closest point the camera measured in that cell in the depth image of that
            step, after the node's filters (DEPTH_MIN, DEPTH_MAX, ground, 11 m histogram radius).
            Needs logs written since 2026-11-16 (vfh.npz "frame").
    memory  closest point in the planner's 30 s point memory, not enlarged (vfh.npz "raw").
    vfh     the histogram 3DVFH+ decides on: every obstacle enlarged by the 1.5 m safety radius,
            so one close point fills many cells. The cells of the obstacles themselves are
            estimated and outlined. The only mode for older logs.

Usage:
    python3 tools/oa/fov_video.py flight_logs/<date>_<time> [--mode frame|memory|vfh]
        [--fov H V] [--out file.mp4] [--frames t1 t2 ...]
Output: <folder>/fov_<mode>.mp4 (H.264, 1280x720), or PNGs with --frames.
Needs matplotlib and gst-launch-1.0 with x264enc and mp4mux (the PC, not the Orin).
Field of view: 90 x 60 deg (Orbbec Gemini 336L depth 640x360) unless the folder has trial.json
(SITL trial: x500_depth_down camera, 73 x 58 deg).

Last Modified: 2026-11-16
"""
import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALPHA, N_EL, N_AZ = 6.0, 30, 60     # bins of vfh3d.py
SAFETY = 1.5                        # VFH3D safety_radius (m)
NONE = 98.0                         # logged value 99 = no obstacle
DMAX = 11.0                         # VFH3D hist_radius (m)
FPS = 10
CMAP = plt.get_cmap("turbo_r")
WHITE = (1.0, 1.0, 1.0, 1.0)
MODES_VI = {"goal": "bay thẳng tới goal", "early": "né sớm (nhìn 10 m)", "block": "né (nhìn 6 m)",
            "slow": "né (nhìn 3 m)", "blocked": "mọi hướng bị chặn, quay tìm đường"}
TITLES = {"frame": "khoảng cách camera đo được",
          "memory": "khoảng cách trong bộ nhớ 30 s của planner",
          "vfh": "histogram 3DVFH+ (đã nới vùng an toàn 1,5 m)"}
FOOTERS = {
    "frame": "Số trong ô: khoảng cách (m) tới điểm gần nhất camera đo được trong ô đó "
             "ở khung hình này (đã bỏ điểm gần hơn DEPTH_MIN, mặt đất và điểm xa hơn 11 m).\n"
             "Ô trắng: camera không đo được vật cản nào trong ô đó.   "
             "Viền đỏ: gần hơn 1,5 m (vùng an toàn của drone).",
    "memory": "Số trong ô: khoảng cách (m) tới điểm gần nhất planner còn nhớ theo hướng đó "
              "(điểm đo được trong 30 s gần nhất, chưa nới vùng an toàn).\n"
              "Ô trắng: không có điểm nào trong 11 m.   "
              "Viền đỏ: gần hơn 1,5 m (vùng an toàn của drone).",
    "vfh": "Số trong ô: khoảng cách (m) 3DVFH+ dùng để chọn hướng (bộ nhớ 30 s). "
           "Mỗi vật cản được nới thêm vùng an toàn 1,5 m nên con số của nó lan sang các ô "
           "xung quanh.\nÔ viền đen: ô có vật cản thật (ước tính).   "
           "Ô trắng: không có vật cản trong 11 m.",
}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def window_n(d):
    return int(math.ceil(math.degrees(math.asin(min(1.0, SAFETY / d))) / ALPHA))


def window_mask(e, a, n):
    m = np.zeros((N_EL, N_AZ), dtype=bool)
    m[max(0, e - n):min(N_EL, e + n + 1), np.arange(a - n, a + n + 1) % N_AZ] = True
    return m


def obstacle_bins(h):
    """
    Enlarged histogram (inf = none) -> bool mask of the bins the obstacles are in (estimate).

    The bins holding exactly the value d are the visible part of the windows of the obstacles at
    distance d: cover them with as few windows as possible, centred on bins whose whole window
    is <= d. Synthetic test: 97 % of the visible obstacle bins found, misplaced ones mostly 1-2
    bins off; bins hidden by a closer obstacle's window cannot be recovered.
    """
    found = np.zeros(h.shape, dtype=bool)
    for d in np.unique(h[np.isfinite(h)]):
        region = h == d
        below = h <= d * 1.0005
        uncovered = region.copy()
        # float16 rounding can move d across a window size step: try both sizes, larger first
        for n in sorted({window_n(d * 0.999), window_n(d * 1.001)}, reverse=True):
            cands = []
            for e, a in zip(*np.nonzero(region)):
                w = window_mask(e, a, n)
                if np.all(below[w]):
                    cands.append((e, a, w))
            while uncovered.any() and cands:
                ue, ua = np.nonzero(uncovered)
                ce = ue.mean()
                ang = np.radians(ua * ALPHA)
                ca = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean()))
                ca = ca / ALPHA % N_AZ
                best, score = None, None
                for e, a, w in cands:
                    cover = int((w & uncovered).sum())
                    da = min(abs(a - ca), N_AZ - abs(a - ca))
                    s = (cover, -((e - ce) ** 2 + da ** 2))
                    if cover and (score is None or s > score):
                        best, score = (e, a, w), s
                if best is None:
                    break
                found[best[0], best[1]] = True
                uncovered &= ~best[2]
            if not uncovered.any():
                break
    return found


def encode(frame_dir, out):
    pipeline = (f"multifilesrc location={frame_dir}/f%05d.png index=0 "
                f"caps=image/png,framerate={FPS}/1 ! pngdec ! videoconvert ! "
                "video/x-raw,format=I420 ! x264enc speed-preset=medium bitrate=2500 ! "
                f"video/x-h264,profile=main ! mp4mux ! filesink location={out}")
    subprocess.run(["gst-launch-1.0", "-q"] + pipeline.split(), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--mode", choices=("frame", "memory", "vfh"))
    ap.add_argument("--fov", type=float, nargs=2, metavar=("H", "V"), help="field of view (deg)")
    ap.add_argument("--out")
    ap.add_argument("--frames", type=float, nargs="*",
                    help="only write PNG frames at these times (s)")
    args = ap.parse_args()

    folder = os.path.abspath(os.path.expanduser(args.folder))
    name = os.path.basename(folder.rstrip("/"))
    with open(os.path.join(folder, "flight.csv")) as f:
        rows = list(csv.DictReader(f))
    ct = np.array([fnum(r["t"]) for r in rows])
    npz = np.load(os.path.join(folder, "vfh.npz"))
    st = npz["t"]
    if len(st) == 0:
        sys.exit("vfh.npz has no 3DVFH+ steps")
    key = {"frame": "frame", "memory": "raw", "vfh": "hist"}
    mode = args.mode or ("frame" if "frame" in npz.files else "vfh")
    if key[mode] not in npz.files:
        sys.exit(f"vfh.npz of {name} has no '{key[mode]}' (logged since 2026-11-16): "
                 "use --mode vfh")
    data = npz[key[mode]].astype(np.float32)
    data[data >= NONE] = np.inf
    marks = [obstacle_bins(h) for h in data] if mode == "vfh" else [h < SAFETY for h in data]
    sitl = os.path.exists(os.path.join(folder, "trial.json"))
    hfov, vfov = args.fov or ((73.0, 58.0) if sitl else (90.0, 60.0))

    # Grid: azimuth cells centred on the camera axis, elevation cells = the histogram rows
    k_az = int(math.ceil((hfov / 2 - ALPHA / 2) / ALPHA))
    k_el = int(math.ceil(vfov / 2 / ALPHA))
    xs = ALPHA * np.arange(-k_az, k_az + 1)
    ys = ALPHA * (np.arange(-k_el, k_el) + 0.5)
    x0, x1, y0, y1 = xs[0] - ALPHA / 2, xs[-1] + ALPHA / 2, ys[0] - ALPHA / 2, ys[-1] + ALPHA / 2
    el_rows = ((ys + 90.0) // ALPHA).astype(int)
    small = len(xs) * len(ys) > 160

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.71])
    title = fig.text(0.5, 0.955, "", ha="center", fontsize=14, weight="bold")
    sub = fig.text(0.5, 0.905, "", ha="center", fontsize=11)
    fig.text(0.5, 0.022, FOOTERS[mode], ha="center", fontsize=9.5)
    im = ax.imshow(np.ones((len(ys), len(xs), 4)), extent=[x0, x1, y0, y1], origin="lower",
                   interpolation="nearest", aspect="auto")
    for x in np.arange(x0, x1 + 0.1, ALPHA):    # cell borders
        ax.axvline(x, color="0.55", lw=0.6, zorder=2)
    for y in np.arange(y0, y1 + 0.1, ALPHA):
        ax.axhline(y, color="0.55", lw=0.6, zorder=2)
    ax.set_xticks(np.arange(-15 * (x1 // 15), x1 + 0.1, 15))
    ax.set_yticks(np.arange(-15 * (y1 // 15), y1 + 0.1, 15))
    ax.add_patch(plt.Rectangle((-hfov / 2, -vfov / 2), hfov, vfov, fill=False, lw=1.5, ls="--",
                               ec="0.2", zorder=4))
    ax.set_xlabel(f"trái   <-   góc ngang so với trục camera (độ), khung nét đứt = góc nhìn "
                  f"{hfov:.0f} x {vfov:.0f} độ   ->   phải")
    ax.set_ylabel("góc ngẩng (độ)")
    texts = [[ax.text(x, y, "", ha="center", va="center", fontsize=9 if small else 10)
              for x in xs] for y in ys]
    edge = "black" if mode == "vfh" else "red"
    boxes = [[ax.add_patch(plt.Rectangle((x - ALPHA / 2, y - ALPHA / 2), ALPHA, ALPHA, fill=False,
                                         lw=3.0, ec=edge, visible=False, zorder=3))
              for x in xs] for y in ys]
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, DMAX))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015).set_label("khoảng cách (m)")
    banner = fig.text(0.5, 0.49, "", ha="center", va="center", fontsize=16, color="white",
                      weight="bold", zorder=10,
                      bbox=dict(facecolor="#c0392b", alpha=0.85, edgecolor="none"))

    def draw(ft):
        k = max(int(np.searchsorted(st, ft, side="right")) - 1, 0)
        r = rows[int(np.clip(np.searchsorted(ct, ft), 0, len(rows) - 1))]
        heading = fnum(r.get("heading"))
        cols = (np.floor(((xs + heading) % 360.0) / ALPHA).astype(int)) % N_AZ
        h, mk = data[k], marks[k]
        rgba = np.zeros((len(ys), len(xs), 4))
        for i, e in enumerate(el_rows):
            for j, a in enumerate(cols):
                d, marked = h[e, a], bool(mk[e, a])
                if np.isfinite(d):
                    c = CMAP(min(d, DMAX) / DMAX)
                    rgba[i, j] = c
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    texts[i][j].set_text(f"{d:.1f}")
                    texts[i][j].set_color("white" if lum < 0.5 else "black")
                else:
                    rgba[i, j] = WHITE
                    texts[i][j].set_text("")
                texts[i][j].set_fontweight("bold" if marked else "normal")
                boxes[i][j].set_visible(marked)
        im.set_data(rgba)
        title.set_text(f"Chuyến bay {name}: {TITLES[mode]}   t = {ft:5.1f} s")
        hold = ", đang đứng giữ vị trí" if r.get("holding") == "1" else ""
        sub.set_text(f"3DVFH+: {MODES_VI.get(r.get('vfh_mode'), '-')}{hold}   |   "
                     f"mũi drone {heading:.0f}°, hướng goal {fnum(r.get('goal_az')):.0f}°, "
                     f"còn {fnum(r.get('dist_goal')):.1f} m   |   "
                     f"cao {fnum(r.get('height')):.2f} m")
        gap = ft - st[k]
        if r.get("stale") == "1":
            banner.set_text("Mất vị trí từ PX4 (rớt kết nối uXRCE / USB): planner tạm dừng")
        elif gap > 0.35:
            banner.set_text(f"Planner không chạy {gap:.1f} s (mất ảnh depth?)")
        else:
            banner.set_text("")
        banner.set_visible(bool(banner.get_text()))

    if args.frames:
        outdir = os.path.dirname(os.path.abspath(args.out)) if args.out else folder
        for ft in args.frames:
            draw(ft)
            p = os.path.join(outdir, f"fov_{mode}_{ft:05.1f}.png")
            fig.savefig(p)
            print(p)
        return

    out = os.path.abspath(os.path.expanduser(args.out or os.path.join(folder, f"fov_{mode}.mp4")))
    times = np.arange(st[0], ct[-1] + 1e-6, 1.0 / FPS)
    tmp = tempfile.mkdtemp(prefix="fov_video_")
    try:
        for n, ft in enumerate(times):
            draw(ft)
            fig.savefig(os.path.join(tmp, f"f{n:05d}.png"))
        encode(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{out}: {len(times)} frames, {len(times) / FPS:.0f} s ({mode})")


if __name__ == "__main__":
    main()
