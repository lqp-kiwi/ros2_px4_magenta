#!/usr/bin/env python3
"""
Bench check of the 3DVFH+ point memory with the real depth camera, without PX4.

No flight controller link, nothing is published, nothing moves.

The depth images go through the same steps as in obstacle_avoidance_static_offboard: every
DEPTH_STRIDE-th pixel, DEPTH_MIN / DEPTH_MAX, CAM_OFFSET, points lower than GROUND_IGNORE above the
floor dropped, then VFH3D.add_points() with its stability check (vfh3d.py confirm_frames: a point
enters the memory once its cell was seen in several images). The drone is assumed still, level and
facing north, the camera --height m above the floor: keep it still, moving it moves every point.

Every second it prints how many points the images had, how many the stability check left out,
the memory size and the nearest remembered point (also saved as check.txt). At the end it writes
the memory log (vfh.npz, vox_* arrays as in flight logs) for tools/oa/voxel_video.py, plus check_*
arrays (per image: time, points, left out, memory size, nearest distance).

Usage (camera running: ros2 run depth_camera camera --no-cloud), from the workspace root with
the px4_control install sourced (tools/oa/record_voxels.sh does all of it and makes the video):
    python3 tools/oa/depth_filter_check.py [--time S] [--height M] [--out DIR]
Output: <out>/<date>_<time>/vfh.npz, default out depth_checks (not flight_logs: this is no flight).
Stops after --time s (0: no limit), or earlier on Enter, Ctrl+C or SIGTERM
(pkill -TERM -f depth_filter_check), and still writes the log.

Last Modified: 2026-11-20
"""
import argparse
import os
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, CameraInfo

from px4_control.vfh3d import VFH3D
from px4_control.flight_logger import FlightLogger
from px4_control.obstacle_avoidance_static_offboard import (
    DEPTH_TOPIC, CAMERA_INFO_TOPIC, DEPTH_STRIDE, DEPTH_MIN, DEPTH_MAX, CAM_OFFSET, GROUND_IGNORE,
    PLAN_RATE, depth_to_meters)

NEAR = 1.6      # points closer than this are counted apart: where the propellers showed up (m)


class Camera(Node):
    def __init__(self):
        super().__init__("depth_filter_check")
        self.latest = {}
        self.create_subscription(Image, DEPTH_TOPIC, lambda m: self.save("depth", m),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC,
                                 lambda m: self.save("camera_info", m),
                                 qos_profile_sensor_data)

    def save(self, name, msg):
        self.latest[name] = msg


def points_of(msg, info, rays, cam_pos):
    """Depth image -> points (NED) of a level drone facing north (body FRD = NED), as the node."""
    depth = depth_to_meters(msg)[::DEPTH_STRIDE, ::DEPTH_STRIDE]
    if rays is None or rays.shape[:2] != depth.shape:
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        v, u = np.mgrid[0:msg.height:DEPTH_STRIDE, 0:msg.width:DEPTH_STRIDE].astype(np.float64)
        rays = np.stack([np.ones_like(u), (u - cx) / fx, (v - cy) / fy], axis=-1)
    valid = np.isfinite(depth) & (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
    p = rays[valid] * depth[valid][:, None] + CAM_OFFSET + cam_pos
    return p[p[:, 2] < -GROUND_IGNORE], rays    # floor at z = 0, z is down


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--time", type=float, default=30.0,
                    help="how long to record (s), default 30, 0 = until Enter / Ctrl+C / SIGTERM")
    ap.add_argument("--height", type=float, default=0.3,
                    help="camera height above the floor (m), default 0.3 (drone on the floor)")
    ap.add_argument("--out", default="depth_checks", help="output folder, default depth_checks")
    args = ap.parse_args()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Camera()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    vfh = VFH3D()
    log = FlightLogger(args.out)
    pos = np.array([0.0, 0.0, -args.height])
    stats = {k: [] for k in ("t", "points", "left_out", "near", "near_left_out", "memory",
                             "nearest")}
    rays, used, t0 = None, None, time.monotonic()
    lines = []

    def say(text):
        print(text, flush=True)
        lines.append(text)

    # Stop triggers besides Ctrl+C: Enter in the terminal, SIGTERM from another one
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    if sys.stdin.isatty():
        threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    print(f"Waiting for {DEPTH_TOPIC} and {CAMERA_INFO_TOPIC}...")
    try:
        while "depth" not in node.latest or "camera_info" not in node.latest:
            if time.monotonic() - t0 > 15:
                sys.exit("No depth image after 15 s: is the camera node running "
                         "(same ROS_DOMAIN_ID)?")
            time.sleep(0.1)
        say(f"Recording {f'{args.time:.0f} s' if args.time > 0 else 'until stopped'}, keep the "
            "drone still. Enter (or Ctrl+C) stops.")
        t0 = last_print = time.monotonic()
        while not stop.is_set() and (args.time <= 0 or time.monotonic() - t0 < args.time):
            time.sleep(1.0 / PLAN_RATE)
            msg = node.latest["depth"]
            if msg is used:
                continue
            used = msg
            pts, rays = points_of(msg, node.latest["camera_info"], rays, pos)
            now = time.monotonic()
            vfh.add_points(pts, now, pos)
            log.add_voxels(now - t0, pos, 0.0, vfh)
            d = np.linalg.norm(pts - pos, axis=1)
            d_out = np.linalg.norm(vfh.rejected - pos, axis=1)
            d_mem = np.linalg.norm(vfh.mem_pts - pos, axis=1)
            nearest = float(d_mem.min()) if len(d_mem) else np.inf
            for k, v in (("t", now - t0), ("points", len(pts)), ("left_out", len(vfh.rejected)),
                         ("near", int((d < NEAR).sum())),
                         ("near_left_out", int((d_out < NEAR).sum())),
                         ("memory", len(vfh.mem_pts)), ("nearest", nearest)):
                stats[k].append(v)
            if now - last_print >= 1.0:
                last_print = now
                n = [i for i, t in enumerate(stats["t"]) if t > now - t0 - 1.0]
                pts_s = sum(stats["points"][i] for i in n)
                out_s = sum(stats["left_out"][i] for i in n)
                near_s = sum(stats["near"][i] for i in n)
                near_out = sum(stats["near_left_out"][i] for i in n)
                say(f"t {now - t0:5.1f} s | {len(n):2d} images | "
                    f"points/image {pts_s / max(len(n), 1):6.0f} | "
                    f"left out {100.0 * out_s / max(pts_s, 1):5.1f} % | "
                    f"closer than {NEAR} m: {near_s} points, {near_out} left out | "
                    f"memory {len(vfh.mem_pts)} voxels, nearest {nearest:.2f} m")
        if stop.is_set():
            say("Stopped")
    except KeyboardInterrupt:
        say("Stopped (Ctrl+C)")
    finally:
        out = os.path.join(log.dir, "vfh.npz")
        np.savez_compressed(out, **log.voxel_arrays(),
                            **{f"check_{k}": np.array(v) for k, v in stats.items()})
        n = len(stats["t"])
        if n > 1:
            pts_all, out_all = sum(stats["points"]), sum(stats["left_out"])
            say(f"{n} images in {stats['t'][-1] - stats['t'][0]:.1f} s "
                f"({(n - 1) / max(stats['t'][-1] - stats['t'][0], 1e-6):.1f} per second used), "
                f"{100.0 * out_all / max(pts_all, 1):.1f} % of the points left out, "
                f"memory {stats['memory'][-1]} voxels")
        with open(os.path.join(log.dir, "check.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved {out}\nVideo: python3 tools/oa/voxel_video.py {log.dir}")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)   # Python aborts at exit if this thread is still inside ROS
        node.destroy_node()


if __name__ == "__main__":
    main()
