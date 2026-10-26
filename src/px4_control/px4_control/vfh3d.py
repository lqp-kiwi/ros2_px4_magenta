#!/usr/bin/env python3

import math
import numpy as np

"""
Script Name: vfh3d.py
Description: 3DVFH+ obstacle avoidance (no ROS code), used by go_to_gps_point.py.
             All positions and points are in the PX4 local NED frame
             (x = North, y = East, z = Down).

Pipeline (each planning step):
    1. add_points()       : keep a memory of obstacle points (voxel filtered)
    2. build_histogram()  : 2D polar histogram (azimuth x elevation), closest distance per bin
                            + enlarge each obstacle by the drone/safety radius
                            + binary histograms (blocked / free) with hysteresis, one per look-ahead
                              distance (early_dist, block_dist, slow_dist)
    3. choose_direction() : look far first (early_dist) and steer early if a free direction close
                            enough to the goal exists; in clutter look closer (block_dist, slow_dist).
                            Pick the free bin with the lowest cost
                            (goal direction + previous direction + heading + elevation
                             + other side of the goal than the obstacle is being passed on)
                            + speed_factor from the obstacles in front of that direction

Last Modified: 2026-10-22
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
                 hist_radius=11.0,    # only obstacles closer than this are put in the histogram (m),
                                      # >= early_dist + hysteresis
                 early_dist=10.0,     # look this far first, to steer early around obstacles (m)
                 max_detour=75.0,     # ... but only into directions this close to the goal direction (deg)
                 block_dist=6.0,      # otherwise a bin is blocked if an obstacle is closer than this (m)
                 hysteresis=0.5,      # a blocked bin is freed only when the obstacle is this much farther (m)
                 safety_radius=1.5,   # drone radius + safety margin (m)
                 slow_dist=3.0,       # slow down when an obstacle in front is closer than this (m), > safety_radius
                 max_elev=25.0,       # max climb/descent angle of a candidate direction (deg)
                 memory_time=30.0,    # how long obstacle points are remembered (s)
                 voxel=0.2,           # voxel size of the point memory (m)
                 side_hold=10,        # steps the goal direction must stay free before the side is forgotten
                 k_goal=5.0, k_prev=2.0, k_head=1.0, k_elev=30.0,   # cost weights
                 k_side=4.0):         # cost of a bin on the other side of the goal than the one chosen
        self.hist_radius = hist_radius
        self.early_dist = early_dist
        self.max_detour = math.radians(max_detour)
        self.block_dist = block_dist
        self.hysteresis = hysteresis
        self.safety_radius = safety_radius
        self.slow_dist = slow_dist
        self.max_elev = math.radians(max_elev)
        self.memory_time = memory_time
        self.voxel = voxel
        self.k_goal, self.k_prev, self.k_head, self.k_elev = k_goal, k_prev, k_head, k_elev
        self.side_hold, self.k_side = side_hold, k_side
        # Side of the goal direction the drone goes around the obstacle: +1 right, -1 left, 0 none yet.
        # Keeping it stops left/right flip-flopping when more of the obstacle comes into view.
        self.side = 0
        self.goal_free_steps = 0

        self.mem_pts = np.empty((0, 3))     # remembered obstacle points (NED)
        self.mem_t = np.empty(0)            # time each point was seen
        self.raw = np.full((N_EL, N_AZ), np.inf)        # closest distance per bin (not enlarged)
        self.hist = np.full((N_EL, N_AZ), np.inf)       # enlarged histogram: closest distance per bin
        self.blocked = {d: np.zeros((N_EL, N_AZ), dtype=bool)   # binary histogram per look-ahead distance
                        for d in (early_dist, block_dist, slow_dist)}
        self.speed_factor = 1.0             # 0.3 .. 1.0, lower when obstacles in front are close
        self.nearest_ahead = np.inf         # closest obstacle within 45 deg of the flight direction (m)
        # Last decision, for logging: mode = "goal" (goal direction free), "early" / "block" / "slow"
        # (look-ahead distance used to find a free bin) or "blocked"; cost terms of the bin flown into
        self.mode = "blocked"
        self.cost = np.full((N_EL, N_AZ), np.inf)   # total cost per bin, inf = not a candidate
        self.cost_terms = {}

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
        self.raw = raw

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

        # Binary histograms with hysteresis (stops blocked/free flickering)
        for d, blocked in self.blocked.items():
            self.blocked[d] = (hist < d) | (blocked & (hist < d + self.hysteresis))

    def choose_direction(self, pos, goal, prev_dir, heading, allow_up=True, allow_down=True):
        """Returns a unit direction (NED) to fly in, or None if every direction is blocked"""
        self.build_histogram(pos)

        to_goal = goal - pos
        goal_dist = float(np.linalg.norm(to_goal))
        goal_dir = to_goal / max(goal_dist, 1e-6)
        e_goal, a_goal = direction_to_bin(goal_dir)

        # Candidate bins: not too steep, and never lower than the goal: diving under an obstacle
        # (tree canopy, branches) traps the drone, the forward camera can't see what is above it
        goal_el = math.asin(float(np.clip(-goal_dir[2], -1.0, 1.0)))
        allowed = (np.abs(EL_C) <= self.max_elev)[:, None] & np.ones(N_AZ, dtype=bool)
        allowed &= (EL_C + math.radians(ALPHA / 2) >= min(goal_el, 0.0))[:, None]
        if not allow_up:
            allowed &= (EL_C <= 0)[:, None]
        if not allow_down:
            allowed &= (EL_C >= 0)[:, None]

        # 3DVFH+ cost function, term by term
        detour = angle_between(BIN_DIRS, goal_dir)
        heading_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        # Side of every azimuth bin seen from the goal direction: +1 right, -1 left, 0 = goal bin +- 1 bin
        rel_az = (AZ_C - math.atan2(goal_dir[1], goal_dir[0]) + math.pi) % (2 * math.pi) - math.pi
        bin_side = np.where(np.abs(rel_az) <= math.radians(ALPHA), 0, np.sign(rel_az)).astype(int)
        terms = {"goal": self.k_goal * detour,
                 "heading": self.k_head * angle_between(BIN_DIRS, heading_dir),
                 "elev": self.k_elev * np.broadcast_to(np.abs(EL_C)[:, None], detour.shape),
                 "prev": (self.k_prev * angle_between(BIN_DIRS, prev_dir) if prev_dir is not None
                          else np.zeros_like(detour)),
                 "side": self.k_side * np.broadcast_to((self.side != 0) & (bin_side == -self.side), detour.shape)}
        cost = sum(terms.values())

        # Look far first: steer early and smoothly around obstacles, as long as that is not a big
        # detour. In clutter nothing close to the goal direction is free that far (a free corridor
        # sideways would trap the drone), so look closer and use the gaps between the obstacles.
        for mode, dist, max_detour in (("early", self.early_dist, self.max_detour),
                                       ("block", self.block_dist, math.pi), ("slow", self.slow_dist, math.pi)):
            blocked = self.blocked[dist]
            # Goal direction free (or the goal is closer than the obstacle) -> fly straight to it
            if not blocked[e_goal, a_goal] or self.hist[e_goal, a_goal] > goal_dist:
                self.remember("goal", np.where(~blocked & allowed, cost, np.inf), terms, e_goal, a_goal)
                self.goal_free_steps += 1
                if self.goal_free_steps >= self.side_hold:     # obstacle passed: next one may go either way
                    self.side = 0
                self.update_speed_factor(goal_dir)
                return goal_dir
            free = ~blocked & allowed & (detour <= max_detour)
            if free.any():
                break
        else:
            self.remember("blocked", np.full(cost.shape, np.inf), terms, None, None)
            return None

        cost = np.where(free, cost, np.inf)
        e, a = np.unravel_index(np.argmin(cost), cost.shape)
        self.remember(mode, cost, terms, e, a)
        self.goal_free_steps = 0
        if self.side == 0:      # first avoidance step: go around on this side until the obstacle is passed
            self.side = int(bin_side[a])
        # The whole bin is free: inside it, use the elevation closest to the goal. Bin centres are
        # at +-3 deg, so the centre alone would make the drone climb or sink while avoiding.
        lo = math.radians(e * ALPHA - 90.0)
        el = float(np.clip(goal_el, lo, lo + math.radians(ALPHA)))
        direction = np.array([math.cos(el) * math.cos(AZ_C[a]), math.cos(el) * math.sin(AZ_C[a]), -math.sin(el)])
        self.update_speed_factor(direction)
        return direction

    def remember(self, mode, cost, terms, e, a):
        """Keep the last decision for logging"""
        self.mode, self.cost = mode, cost
        self.cost_terms = {k: float(v[e, a]) for k, v in terms.items()} if e is not None else {}

    def update_speed_factor(self, direction):
        """speed_factor 0.3 .. 1.0 from the closest obstacle within 45 deg of the flight direction.
        Obstacles beside or behind the drone (already passed) do not slow it down."""
        ahead = angle_between(BIN_DIRS, direction) < math.radians(45.0)
        self.nearest_ahead = float(self.raw[ahead].min())
        self.speed_factor = float(np.clip((self.nearest_ahead - self.safety_radius) /
                                          (self.slow_dist - self.safety_radius), 0.3, 1.0))
