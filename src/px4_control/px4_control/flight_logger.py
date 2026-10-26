#!/usr/bin/env python3

import csv
import math
import os
import sys
import time

import numpy as np

"""
Script Name: flight_logger.py
Description: Records a flight of obstacle_avoidance_static_gps.py for analysis (no ROS code).
             One folder per flight, <log dir>/<date>_<time>/:
                 flight.csv      one row per log step: position, velocity, setpoints, GPS, altitude (EKF)
                                 and height above the ground (distance sensor), 3DVFH+ decision and
                                 cost terms, distances to obstacles, ...
                 vfh.npz         every planning step: polar histogram, cost map, blocked bins,
                                 goal / chosen direction; all obstacle points seen (0.3 m grid)
                 path.png        top and side view of the path over the obstacles
                 timeseries.png  distance, speed, altitude, yaw, cost terms, acceleration over time
                 decision_*.png  polar histogram and cost map at the first avoidance step
                                 and at the closest obstacle
             All positions are PX4 local NED (north, east, down), angles in degrees.

Re-plot a saved flight:
    python3 -m px4_control.flight_logger flight_logs/<date>_<time>

Last Modified: 2026-10-22
"""

MAP_VOXEL = 0.3     # grid of the obstacle map (m)
MODE_COLORS = {"early": "gold", "block": "orange", "slow": "red", "blocked": "black"}


class FlightLogger:
    def __init__(self, root):
        self.dir = os.path.join(root, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.file = None            # flight.csv, opened at the first row (columns = keys of that row)
        self.closed = False         # set by save(): rows logged after that are dropped
        self.voxels = set()         # obstacle map
        self.steps = {k: [] for k in ("t", "pos", "hist", "cost", "blocked", "goal_dir", "direction")}

    def log(self, row):
        if self.closed:
            return
        if self.file is None:
            self.file = open(os.path.join(self.dir, "flight.csv"), "w", newline="")
            self.writer = csv.DictWriter(self.file, list(row))
            self.writer.writeheader()
        # 4 decimals (0.1 mm, 0.0001 deg), but GPS coordinates need 8 (1 mm)
        self.writer.writerow({k: round(v, 8 if k in ("lat", "lon") else 4) if isinstance(v, float) else v
                              for k, v in row.items()})
        self.file.flush()

    def add_obstacles(self, points):
        """Obstacle points seen by the camera (NED), kept on a MAP_VOXEL grid for the map"""
        if len(points):
            self.voxels.update(map(tuple, np.unique(np.floor(points / MAP_VOXEL).astype(np.int32), axis=0)))

    def add_step(self, t, pos, vfh, goal_dir, direction):
        """3DVFH+ state of one planning step. pos = (north, east, altitude above take-off),
        blocked: bit 0 = early_dist, 1 = block_dist, 2 = slow_dist"""
        blocked = np.zeros(vfh.hist.shape, dtype=np.uint8)
        for bit, d in enumerate((vfh.early_dist, vfh.block_dist, vfh.slow_dist)):
            blocked |= vfh.blocked[d].astype(np.uint8) << bit
        self.steps["t"].append(t)
        self.steps["pos"].append(np.asarray(pos, dtype=np.float32))
        self.steps["hist"].append(np.minimum(vfh.hist, 99.0).astype(np.float16))     # 99 = no obstacle
        self.steps["cost"].append(np.minimum(vfh.cost, 999.0).astype(np.float16))    # 999 = not a candidate
        self.steps["blocked"].append(blocked)
        self.steps["goal_dir"].append(np.asarray(goal_dir, dtype=np.float32))
        self.steps["direction"].append(np.asarray(direction if direction is not None else [np.nan] * 3,
                                                  dtype=np.float32))

    def save(self):
        """Write vfh.npz and the plots, return the folder"""
        self.closed = True
        if self.file is not None:
            self.file.close()
            self.file = None
        obstacles = (np.array(sorted(self.voxels), dtype=np.float32).reshape(-1, 3) + 0.5) * MAP_VOXEL
        np.savez_compressed(os.path.join(self.dir, "vfh.npz"), obstacles=obstacles,
                            **{k: np.array(v) for k, v in self.steps.items()})
        try:
            plot(self.dir)
        except Exception as e:      # never lose the data because of a plot
            print(f"Plotting failed ({e}), data is in {self.dir}")
        return self.dir


def read_csv(path):
    """flight.csv -> dict of numpy arrays (text columns stay strings)"""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def shade_modes(ax, t, mode):
    """Background colour where 3DVFH+ had to avoid an obstacle"""
    for name, color in MODE_COLORS.items():
        on = mode == name
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[i0], t[min(i1, len(t) - 1)], color=color, alpha=0.25, lw=0)


def plot(folder):
    import warnings
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")   # harmless, 2 matplotlib installs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    f = read_csv(os.path.join(folder, "flight.csv"))
    npz = np.load(os.path.join(folder, "vfh.npz"))
    t = f["t"]
    obs = npz["obstacles"]
    obs = obs[(-obs[:, 2] > 0.3) & (-obs[:, 2] < 12)] if len(obs) else obs
    flying = f["phase"] != "wait"

    # ---------- path: top and side view ----------
    fig, (top, side) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 1]})
    if len(obs):
        top.scatter(obs[:, 1], obs[:, 0], s=2, c="0.6", label="obstacle points seen")
        side.scatter(obs[:, 1], -obs[:, 2], s=2, c="0.6")
    e, n, alt, v = f["east"][flying], f["north"][flying], f["alt"][flying], f["speed"][flying]
    seg = np.stack([np.c_[e[:-1], n[:-1]], np.c_[e[1:], n[1:]]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidth=2.5)
    lc.set_array(v[:-1])
    top.add_collection(lc)
    fig.colorbar(lc, ax=top, label="horizontal speed (m/s)")
    mode = f["vfh_mode"][flying]
    for name, color in MODE_COLORS.items():
        m = mode == name
        if m.any():
            top.scatter(e[m], n[m], s=10, c=color, label=f"avoiding: {name}", zorder=3)
    top.plot(e[0], n[0], "g^", ms=12, label="start")
    top.plot(f["goal_e"][0], f["goal_n"][0], "r*", ms=16, label="goal")
    top.set_aspect("equal")
    top.autoscale()
    top.set_xlabel("east (m)")
    top.set_ylabel("north (m)")
    top.set_title(f"Flight {os.path.basename(folder)}: {np.sum(np.hypot(np.diff(e), np.diff(n))):.1f} m, "
                  f"{t[flying][-1] - t[flying][0]:.0f} s")
    top.legend(loc="best", fontsize=8)
    top.grid(alpha=0.3)
    side.plot(e, alt, "b-", label="altitude (EKF, above the take-off point)")
    if "agl" in f:
        side.plot(e, f["agl"][flying], "r-", label="height above the ground (distance sensor)")
    side.axhline(f["alt_target"][0], color="k", ls="--", lw=1, label="flight altitude")
    side.set_xlabel("east (m)")
    side.set_ylabel("m")
    side.legend(fontsize=8)
    side.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "path.png"), dpi=120)
    plt.close(fig)

    # ---------- time series ----------
    fig, ax = plt.subplots(6, 1, figsize=(12, 17), sharex=True)
    phase = f["phase"]
    changes = [i for i in range(1, len(t)) if phase[i] != phase[i - 1]]
    for a in ax:
        for i in changes:
            a.axvline(t[i], color="k", lw=0.8, ls=":")
        a.grid(alpha=0.3)
    for i in [0] + changes:
        ax[0].text(t[i], 1.02, phase[i], transform=ax[0].get_xaxis_transform(), fontsize=8)
    ax[0].plot(t, f["dist_goal"], label="distance to goal")
    ax[0].plot(t, np.minimum(f["nearest_ahead"], 15), label="nearest obstacle ahead (+-45 deg)")
    ax[0].plot(t, np.minimum(f["nearest_all"], 15), label="nearest obstacle (all directions)", alpha=0.6)
    ax[0].set_ylabel("m")
    ax[1].plot(t, np.linalg.norm(np.c_[f["vel_target_n"], f["vel_target_e"], f["vel_target_d"]], axis=1),
               label="planner target")
    ax[1].plot(t, np.linalg.norm(np.c_[f["vel_cmd_n"], f["vel_cmd_e"], f["vel_cmd_d"]], axis=1),
               label="smoothed setpoint sent to PX4")
    ax[1].plot(t, np.linalg.norm(np.c_[f["vn"], f["ve"], f["vd"]], axis=1), label="actual")
    ax1b = ax[1].twinx()
    ax1b.plot(t, f["speed_factor"], "k:", label="speed factor")
    ax1b.set_ylim(0, 1.1)
    ax1b.set_ylabel("speed factor")
    ax[1].set_ylabel("speed (m/s)")
    ax[2].plot(t, f["alt"], label="altitude (EKF, above the take-off point)")
    if "agl" in f:
        ax[2].plot(t, f["agl"], label="height above the ground (distance sensor)")
    ax[2].plot(t, f["alt_target"], "k--", label="flight altitude")
    ax[2].set_ylabel("m")
    for k, label in (("heading", "heading"), ("yaw_cmd", "yaw setpoint (smoothed)"),
                     ("dir_az", "3DVFH+ direction"), ("goal_az", "goal direction")):
        ax[3].plot(t, np.degrees(np.unwrap(np.radians(np.nan_to_num(f[k])))), label=label,
                   lw=1 if k != "heading" else 2)
    ax[3].set_ylabel("azimuth (deg)")
    for k in ("cost_goal", "cost_heading", "cost_elev", "cost_prev", "cost_side", "cost_total"):
        ax[4].plot(t, f[k], label=k[5:], lw=2 if k == "cost_total" else 1)
    shade_modes(ax[4], t, f["vfh_mode"])
    ax[4].set_ylabel("cost of the chosen bin")
    ax[4].set_title("background: avoiding with look-ahead " +
                    ", ".join(f"{k} = {c}" for k, c in MODE_COLORS.items()), fontsize=8)
    ax[5].plot(t, f["acc_cmd"], label="|acceleration setpoint|")
    ax[5].plot(t, np.linalg.norm(np.c_[f["an"], f["ae"], f["ad"]], axis=1), label="|acceleration| (EKF)", alpha=0.7)
    ax[5].set_ylabel("m/s2")
    ax[5].set_xlabel("time (s)")
    for a in list(ax) + [ax1b]:
        a.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "timeseries.png"), dpi=110)
    plt.close(fig)

    # ---------- 3DVFH+ decisions ----------
    if len(npz["t"]) == 0:
        return
    hist = npz["hist"].astype(np.float32)
    avoiding = np.flatnonzero(np.isfinite(npz["direction"][:, 0]) &
                              (np.linalg.norm(npz["direction"] - npz["goal_dir"], axis=1) > 1e-3))
    picks = {}
    if len(avoiding):
        picks["first_avoidance"] = avoiding[0]
    picks["closest_obstacle"] = int(np.argmin(hist.reshape(len(hist), -1).min(axis=1)))
    for name, i in picks.items():
        plot_decision(plt, folder, name, npz, i)


def plot_decision(plt, folder, name, npz, i):
    def az_el(d):
        return math.degrees(math.atan2(d[1], d[0])) % 360, math.degrees(math.asin(np.clip(-d[2], -1, 1)))

    hist = npz["hist"][i].astype(np.float32)
    cost = npz["cost"][i].astype(np.float32)
    blocked = npz["blocked"][i]
    ext = [0, 360, -90, 90]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    im = a0.imshow(np.where(hist < 99, hist, np.nan), origin="lower", extent=ext, aspect="auto", cmap="inferno_r")
    fig.colorbar(im, ax=a0, label="closest obstacle, enlarged by safety radius (m)")
    for bit, (ls, label) in enumerate(((":", "early_dist"), ("-", "block_dist"))):
        a0.contour((blocked >> bit) & 1, levels=[0.5], extent=ext, colors="cyan", linestyles=ls)
        a0.plot([], [], "c" + ls, label=f"blocked at {label}")
    im = a1.imshow(np.where(cost < 999, cost, np.nan), origin="lower", extent=ext, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=a1, label="cost (candidate bins only)")
    for a in (a0, a1):
        a.plot(*az_el(npz["goal_dir"][i]), "r*", ms=14, label="goal")
        if np.isfinite(npz["direction"][i][0]):
            a.plot(*az_el(npz["direction"][i]), "wo", mec="k", ms=10, label="chosen")
        a.set_xlabel("azimuth (deg, 0 = north, 90 = east)")
        a.set_ylabel("elevation (deg)")
        a.set_ylim(-60, 60)
        a.legend(loc="lower right", fontsize=8)
    p = npz["pos"][i]
    fig.suptitle(f"{name.replace('_', ' ')}: t = {npz['t'][i]:.1f} s, north {p[0]:.1f} m, east {p[1]:.1f} m, "
                 f"altitude {p[2]:.1f} m")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"decision_{name}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1])
    print(f"Plots written to {sys.argv[1]}")
