#!/usr/bin/env python3

import math
import numpy as np

"""
Script Name: vfh3d.py
Description: 3DVFH+ obstacle avoidance (no ROS code), used by go_to_gps_point.py.
             All positions and points are in the PX4 local NED frame
             (x = North, y = East, z = Down).

Pipeline (each planning step):
    1. add_points()       : keep a short memory of obstacle points (voxel filtered)
    2. build_histogram()  : 2D polar histogram (azimuth x elevation), closest distance per bin
                            + enlarge each obstacle by the drone/safety radius
                            + binary histogram (blocked / free) with hysteresis
    3. choose_direction() : pick the free bin with the lowest cost
                            (goal direction + previous direction + heading + elevation)

Last Modified: 2026-10-10
"""

ALPHA = 6.0                    # histogram bin size (degrees)
N_AZ = int(360 / ALPHA)        # azimuth bins: 0 deg = North, 90 deg = East
N_EL = int(180 / ALPHA)        # elevation bins: -90 deg (down) .. +90 deg (up)

# Unit vector (NED) at the centre of every bin, shape (N_EL, N_AZ, 3)
AZ_C = np.radians((np.arange(N_AZ) + 0.5) * ALPHA)
EL_C = np.radians((np.arange(N_EL) + 0.5) * ALPHA - 90.0)
_el, _az = np.meshgrid(EL_C, AZ_C, indexing="ij")
BIN_DIRS = np.stack([np.cos(_el) * np.cos(_az),
                     np.cos(_el) * np.sin(_az),
                     -np.sin(_el)], axis=-1)


def angle_between(dirs, v):
    """Angle (rad) between every unit vector in dirs and the unit vector v"""
    return np.arccos(np.clip(dirs @ v, -1.0, 1.0))


def direction_to_bin(d):
    """Unit vector (NED) -> (elevation index, azimuth index)"""
    az = math.degrees(math.atan2(d[1], d[0])) % 360.0
    el = math.degrees(math.asin(float(np.clip(-d[2], -1.0, 1.0))))
    return min(int((el + 90.0) // ALPHA), N_EL - 1), int(az // ALPHA) % N_AZ


class VFH3D:
    def __init__(self,
                 hist_radius=6.0,     # only obstacles closer than this are put in the histogram (m)
                 block_dist=3.0,      # a bin is blocked if an obstacle is closer than this (m)
                 hysteresis=0.5,      # a blocked bin is freed only when the obstacle is block_dist + this away (m)
                 safety_radius=1.0,   # drone radius + safety margin (m)
                 max_elev=25.0,       # max climb/descent angle of a candidate direction (deg)
                 memory_time=5.0,     # how long obstacle points are remembered (s)
                 voxel=0.2,           # voxel size of the point memory (m)
                 k_goal=5.0, k_prev=2.0, k_head=1.0, k_elev=2.0):  # cost weights
        self.hist_radius = hist_radius
        self.block_dist = block_dist
        self.hysteresis = hysteresis
        self.safety_radius = safety_radius
        self.max_elev = math.radians(max_elev)
        self.memory_time = memory_time
        self.voxel = voxel
        self.k_goal, self.k_prev, self.k_head, self.k_elev = k_goal, k_prev, k_head, k_elev

        self.mem_pts = np.empty((0, 3))     # remembered obstacle points (NED)
        self.mem_t = np.empty(0)            # time each point was seen
        self.hist = np.full((N_EL, N_AZ), np.inf)       # enlarged histogram: closest distance per bin
        self.blocked = np.zeros((N_EL, N_AZ), dtype=bool)
        self.speed_factor = 1.0             # 0.3 .. 1.0, lower when obstacles are close

    def add_points(self, points, stamp, pos):
        # Newest points first, so np.unique keeps the newest point of each voxel
        self.mem_pts = np.vstack([points, self.mem_pts])
        self.mem_t = np.concatenate([np.full(len(points), stamp), self.mem_t])

        # Forget old points and points far away from the drone
        keep = (stamp - self.mem_t < self.memory_time) & \
               (np.linalg.norm(self.mem_pts - pos, axis=1) < 2 * self.hist_radius)
        self.mem_pts, self.mem_t = self.mem_pts[keep], self.mem_t[keep]

        # Voxel filter: one point per voxel
        if len(self.mem_pts):
            keys = np.floor(self.mem_pts / self.voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            self.mem_pts, self.mem_t = self.mem_pts[idx], self.mem_t[idx]

    def build_histogram(self, pos):
        rel = self.mem_pts - pos
        dist = np.linalg.norm(rel, axis=1)
        m = (dist > 0.1) & (dist < self.hist_radius)
        rel, dist = rel[m], dist[m]

        # Polar histogram: closest obstacle distance in every bin
        az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
        el = np.degrees(np.arcsin(np.clip(-rel[:, 2] / dist, -1.0, 1.0)))
        a = (az // ALPHA).astype(int) % N_AZ
        e = np.clip(((el + 90.0) // ALPHA).astype(int), 0, N_EL - 1)
        raw = np.full((N_EL, N_AZ), np.inf)
        np.minimum.at(raw, (e, a), dist)

        # Enlarge every obstacle by the safety radius: closer obstacle -> wider angle
        hist = raw.copy()
        for ei, ai in zip(*np.nonzero(np.isfinite(raw))):
            d = raw[ei, ai]
            lam = math.degrees(math.asin(min(1.0, self.safety_radius / d)))
            n = int(math.ceil(lam / ALPHA))
            rows = slice(max(0, ei - n), min(N_EL, ei + n + 1))
            cols = np.arange(ai - n, ai + n + 1) % N_AZ
            hist[rows, cols] = np.minimum(hist[rows, cols], d)
        self.hist = hist

        # Binary histogram with hysteresis (stops blocked/free flickering)
        self.blocked = (hist < self.block_dist) | \
                       (self.blocked & (hist < self.block_dist + self.hysteresis))

    def choose_direction(self, pos, goal, prev_dir, heading, allow_up=True, allow_down=True):
        """Returns a unit direction (NED) to fly in, or None if every direction is blocked"""
        self.build_histogram(pos)

        to_goal = goal - pos
        goal_dist = float(np.linalg.norm(to_goal))
        goal_dir = to_goal / max(goal_dist, 1e-6)

        nearest = float(self.hist.min())
        self.speed_factor = float(np.clip((nearest - self.safety_radius) /
                                          (self.block_dist - self.safety_radius), 0.3, 1.0))

        # Goal direction free (or the goal is closer than the obstacle) -> fly straight to it
        e, a = direction_to_bin(goal_dir)
        if not self.blocked[e, a] or self.hist[e, a] > goal_dist:
            return goal_dir

        # Candidate bins: free and not too steep
        free = ~self.blocked & (np.abs(EL_C) <= self.max_elev)[:, None]
        if not allow_up:
            free &= (EL_C <= 0)[:, None]
        if not allow_down:
            free &= (EL_C >= 0)[:, None]
        if not free.any():
            return None

        # 3DVFH+ cost function
        heading_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        cost = self.k_goal * angle_between(BIN_DIRS, goal_dir) \
            + self.k_head * angle_between(BIN_DIRS, heading_dir) \
            + self.k_elev * np.abs(EL_C)[:, None]
        if prev_dir is not None:
            cost += self.k_prev * angle_between(BIN_DIRS, prev_dir)
        cost[~free] = np.inf

        e, a = np.unravel_index(np.argmin(cost), cost.shape)
        return BIN_DIRS[e, a]