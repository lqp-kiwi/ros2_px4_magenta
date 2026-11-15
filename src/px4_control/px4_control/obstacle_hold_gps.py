#!/usr/bin/env python3

import sys
import math
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, CameraInfo
from px4_msgs.msg import (VehicleGlobalPosition, VehicleLocalPosition, VehicleAttitude, VehicleStatus, VehicleCommand,
                          OffboardControlMode, TrajectorySetpoint)

from px4_control.flight_logger import FlightLogger
from px4_control.gps_point import load_gps_point

"""
Script Name: obstacle_hold_gps.py
Description: Takes off, flies in a straight line to the GPS point saved in a .json file by record_drone_state.py
             and lands there. Obstacles are not avoided: when the depth camera sees one on the way, or has no image,
             the drone brakes and holds its position. It goes on by itself once the way has been clear for
             CLEAR_TIME. While an obstacle stays the drone keeps holding, until Ctrl+C leaves it in PX4 Hold.
             The camera node (depth_camera package) must already be running.
             The whole flight is in OFFBOARD: from before arming until PX4 disarms after landing, the node streams
             velocity setpoints over uXRCE-DDS, the link of MicroXRCEAgent is the only one it needs.
             A radio keeps working: switching PX4 to another mode ends the node, the drone is then the pilot's.

PX4 parameters:
    COM_OF_LOSS_T        OFFBOARD is lost this long after the last setpoint (default 1 s), e.g. when the node dies.
    COM_OBL_RC_ACT       what PX4 then does. Ctrl+C does not trigger it: the node switches to Hold before it ends.

Last Modified: 2026-11-09

Usage:
    ros2 run px4_control obstacle_hold_gps              the point saved by record_drone_state (gps_point.py)
    ros2 run px4_control obstacle_hold_gps filename     the first point in this .json file
"""

# PX4 topics to read: name -> (topic, message type)
TOPICS = {
    "global_position": ("/fmu/out/vehicle_global_position", VehicleGlobalPosition),
    "local_position": ("/fmu/out/vehicle_local_position", VehicleLocalPosition),
    "attitude": ("/fmu/out/vehicle_attitude", VehicleAttitude),
    "status": ("/fmu/out/vehicle_status", VehicleStatus),
}
DEPTH_TOPIC = "/camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/camera/depth/camera_info"

MAX_EPH = 3.0             # max horizontal position error (m) before flying is allowed

# Obstacles: when to hold
HOLD_DIST = 3.0           # hold when an obstacle on the way is closer than this (m). Braking from MAX_SPEED takes
                          # 0.3 m and 2 s (SITL), so the drone stops about 0.5 m closer. Raise it with MAX_SPEED
CORRIDOR_RADIUS = 1.5     # on the way = closer than this to the straight line to the goal (m), drone radius + margin
MIN_OBSTACLE_POINTS = 5   # fewer depth points than this on the way count as noise
CLEAR_TIME = 2.0          # go on when the way has been clear for this long (s)

# Flight
FLIGHT_ALT = 2.0          # flight height above the ground below the drone, measured by the downward distance
                          # sensor; above the take-off point (EKF) while the sensor has no valid reading (m)
USE_GOAL_ALT = False      # True: fly at the altitude saved in the .json instead of FLIGHT_ALT
MIN_ALT = 1.5             # with USE_GOAL_ALT: never lower than this above the take-off point (m)
ALT_GAIN = 0.5            # vertical speed per meter of height error when holding FLIGHT_ALT (1/s)
MAX_CLIMB = 0.3           # max vertical speed when holding FLIGHT_ALT (m/s)
ACCEPT_RADIUS = 1.5       # closer than this horizontally: straight to the center of the goal (m)
LAND_RADIUS = 0.15        # start landing when closer than this to the center and slow (m)
APPROACH_SPEED = 0.3      # max horizontal speed to the center (m/s)
APPROACH_GAIN = 0.4       # horizontal speed per meter from the center (1/s)
APPROACH_TIMEOUT = 60.0   # land anyway when not settled over the center after flying this long (s)
SETTLE_TIME = 1.0         # stay within LAND_RADIUS this long before landing (s)
LAND_FIX_HEIGHT = 0.5     # no more horizontal corrections below this height, for a straight touchdown (m)
MAX_SPEED = 0.3           # limit of the total speed, horizontal and vertical together (m/s)
SPEED_GUARD = 0.45        # real speed above this: zero velocity is commanded (m/s), a last resort
LPOS_TIMEOUT = 0.5        # no local position for this long: zero velocity too, the node does not know where it is (s)
SLOW_DOWN_DIST = 1.5     # start slowing down this far from the goal (m)
YAW_LOCK_DIST = 2.0       # closer than this to the goal: stop turning towards the goal (m)
MAX_HEADING_ERR = math.radians(20)  # only move when the camera looks within this angle of where the drone goes
FULL_SPEED_HEADING_ERR = math.radians(10)  # full speed below this heading error, slower up to MAX_HEADING_ERR
LOOP_RATE = 10.0          # flight loop and obstacle checks per second
SETPOINT_RATE = 20.0      # OFFBOARD setpoints per second (PX4 leaves OFFBOARD without them, COM_OF_LOSS_T)
STOP_SPEED = 0.02         # below this and not asked to move: the position is held instead of a velocity of 0 (m/s)
TAKEOFF_SPEED = 0.3       # climb speed during take-off (m/s)
LAND_SPEED = 0.3         # descent speed when landing (m/s), PX4 needs > 1.1 * LNDMC_Z_VEL_MAX to detect the landing
HOLD_TIMEOUT = 10.0       # time to get PX4 to stay in Hold after Ctrl+C (s)
HOVER_TIME = 1.0          # hover after take-off before flying forward, so the climb has fully stopped (s)

# Setpoint smoothing: a step in velocity or yaw makes the drone jerk (shakes the camera, can crash a
# real drone). The velocity setpoint is jerk-limited, the yaw setpoint acceleration-limited. Low
# values = smooth flight (tilt ~ MAX_ACC / g), but a longer way to brake before an obstacle.
# They limit how fast the speed changes, not the speed itself: the setpoint never goes above its target.
MAX_ACC = 0.5           # (m/s^2)
MAX_JERK = 0.5            # (m/s^3)
VEL_TAU = 1.0             # time constant to reach a new velocity (s), >= MAX_ACC / MAX_JERK
MAX_YAW_RATE = math.radians(30)   # (rad/s)
MAX_YAW_ACC = math.radians(30)    # (rad/s^2)
YAW_TAU = 1.0             # (s), >= MAX_YAW_RATE / MAX_YAW_ACC

# Flight log (flight_logger.py): one folder per flight with flight.csv, vfh.npz and plots
LOG_DIR = "flight_logs"   # relative to the folder the script is started from
LOG_RATE = 10.0           # rows per second

# Depth camera
DEPTH_STRIDE = 8          # use every 8th pixel
DEPTH_MIN = 1.0           # (m) nearer pixels are ignored: the two front propellers are in the image, on the real
                          # drone they gave depth readings up to at least 0.53 m
DEPTH_MAX = 12.0          # (m)
DEPTH_TIMEOUT = 0.5       # hold position if no depth image for this long (s)
CAM_OFFSET = np.array([0.13, 0.0, -0.02])  # camera position in body FRD (m), at the front of the drone with the two
                                           # front propellers in its field of view; camera faces forward, level
GROUND_IGNORE = 1.0       # ignore points lower than this above the ground (below the drone, else take-off point) (m)
                          # so the drone passes over grass: tips at 0.6 m look 0.3 m higher with a camera tilted 5 deg

EARTH_RADIUS = 6371000.0


def versioned(topic, msg_type):
    """PX4 v1.16+ adds _v<MESSAGE_VERSION> to the topic name of versioned messages
    (e.g. vehicle_local_position_v1). Older PX4 uses the plain name."""
    version = getattr(msg_type, "MESSAGE_VERSION", 0)
    return f"{topic}_v{version}" if version else topic


def gps_to_ned(lat, lon, ref_lat, ref_lon):
    """GPS -> local North/East (m), same azimuthal equidistant projection as PX4"""
    lat, lon, ref_lat, ref_lon = map(math.radians, (lat, lon, ref_lat, ref_lon))
    cos_d_lon = math.cos(lon - ref_lon)
    arg = math.sin(ref_lat) * math.sin(lat) + math.cos(ref_lat) * math.cos(lat) * cos_d_lon
    c = math.acos(max(-1.0, min(1.0, arg)))
    k = 1.0 if abs(c) < 1e-9 else c / math.sin(c)
    north = k * (math.cos(ref_lat) * math.sin(lat) - math.sin(ref_lat) * math.cos(lat) * cos_d_lon) * EARTH_RADIUS
    east = k * math.cos(lat) * math.sin(lon - ref_lon) * EARTH_RADIUS
    return north, east


def quat_to_rot(q):
    """PX4 quaternion [w, x, y, z] (body FRD -> NED) -> rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def depth_to_meters(msg):
    """Depth image -> 2D numpy array in meters (16UC1/mono16 = mm, 32FC1 = m)"""
    if msg.encoding in ("16UC1", "mono16"):
        dtype = ">u2" if msg.is_bigendian else "<u2"
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return img.astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        dtype = ">f4" if msg.is_bigendian else "<f4"
        return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 4)[:, :msg.width]
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def clip_norm(v, max_norm):
    n = float(np.linalg.norm(v))
    return v * (max_norm / n) if n > max_norm else v


class SetpointSmoother:
    """Velocity setpoint with limited acceleration and jerk, yaw with limited rate and acceleration:
    whatever the flight loop asks for, the setpoints sent to PX4 never jump"""

    def __init__(self):
        self.reset(0.0)

    def reset(self, yaw):
        self.vel, self.acc = np.zeros(3), np.zeros(3)
        self.yaw, self.yaw_rate = yaw, 0.0

    def update(self, vel_target, yaw_target, dt):
        acc_target = clip_norm((np.asarray(vel_target, dtype=float) - self.vel) / VEL_TAU, MAX_ACC)
        self.acc = self.acc + clip_norm(acc_target - self.acc, MAX_JERK * dt)
        self.vel = self.vel + self.acc * dt
        rate_target = float(np.clip(wrap_pi(yaw_target - self.yaw) / YAW_TAU, -MAX_YAW_RATE, MAX_YAW_RATE))
        self.yaw_rate += float(np.clip(rate_target - self.yaw_rate, -MAX_YAW_ACC * dt, MAX_YAW_ACC * dt))
        self.yaw = wrap_pi(self.yaw + self.yaw_rate * dt)


class ObstacleHoldGPS(Node):
    def __init__(self):
        super().__init__("obstacle_hold_gps")
        self.latest = {}    # newest message of each topic
        self.stamp = {}     # time each topic was last received
        # PX4 publishes best-effort, so the subscriber QoS must match
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        for name, (topic, msg_type) in TOPICS.items():
            # Listen to both the plain and the versioned name: works with old and new PX4
            for t in {topic, versioned(topic, msg_type)}:
                self.create_subscription(msg_type, t, lambda msg, name=name: self.save_msg(name, msg), qos)
        # PX4 v1.16 names it vehicle_status_v1 (px4_msgs now has a newer VehicleStatus version)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1",
                                 lambda msg: self.save_msg("status", msg), qos)
        self.create_subscription(Image, DEPTH_TOPIC, lambda msg: self.save_msg("depth", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, lambda msg: self.save_msg("camera_info", msg),
                                 qos_profile_sensor_data)

        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)

        # Current velocity target, streamed by the timer through the smoother. None = nothing is streamed
        self.setpoint = None
        self.smooth = SetpointSmoother()
        self.lock_xy = None     # (x, y) held while stopped, instead of a velocity of 0
        self.lock_z = None      # z held while stopped in the air
        self.guarding = False   # True while zero velocity is commanded by SPEED_GUARD / LPOS_TIMEOUT
        self.create_timer(1.0 / SETPOINT_RATE, self.publish_setpoints)

        self.start_z = 0.0      # local z of the take-off point
        self.goal = None        # goal in local NED
        self.holding = False    # True while stopped because of an obstacle / no camera
        self.hold_reason = ""   # why, for the status prints
        self.clear_since = None     # time the way became clear while holding
        self.nearest = math.inf     # distance to the obstacle on the way, inf = none, nan = no camera image
        self.nearest_all = math.inf     # distance to the closest point in the camera image
        self.height_source = None   # "sensor" or "ekf", to print when it changes
        self.rays = None        # pixel rays (body FRD) for the depth image

        # Flight log
        self.logger = None      # FlightLogger while flying
        self.phase = "wait"     # wait, arming, takeoff, hover, cruise, approach, landing
        self.t0 = time.monotonic()
        self.create_timer(1.0 / LOG_RATE, self.log_row)

    def save_msg(self, name, msg):
        self.latest[name] = msg
        self.stamp[name] = time.monotonic()

    # ---------- state ----------
    def data_ready(self):
        if any(n not in self.latest for n in list(TOPICS) + ["depth", "camera_info"]):
            return False
        gp, lp = self.latest["global_position"], self.latest["local_position"]
        return gp.eph < MAX_EPH and not gp.dead_reckoning and lp.xy_valid and lp.z_valid and lp.xy_global

    def is_armed(self):
        return self.latest["status"].arming_state == VehicleStatus.ARMING_STATE_ARMED

    def in_offboard(self):
        return self.latest["status"].nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def position(self):
        lp = self.latest["local_position"]
        return np.array([lp.x, lp.y, lp.z]), lp.heading

    def height(self):
        """(height, local z of the ground, source): above the ground below the drone from the downward
        distance sensor (EKF2 dist_bottom) while it has a valid reading, else above the take-off point (EKF)"""
        lp = self.latest["local_position"]
        if lp.dist_bottom_valid and lp.dist_bottom_sensor_bitfield & VehicleLocalPosition.DIST_BOTTOM_SENSOR_RANGE:
            return lp.dist_bottom, lp.z + lp.dist_bottom, "sensor"
        return self.start_z - lp.z, self.start_z, "ekf"

    def target_z(self, goal):
        """Local z (NED) of the flight altitude: FLIGHT_ALT above the ground (or the altitude of the .json)"""
        return goal[2] if USE_GOAL_ALT else self.height()[1] - FLIGHT_ALT

    # ---------- commands ----------
    def publish_setpoints(self):
        """OFFBOARD stream (timer): the smoothed velocity, or the position where the drone stopped. No acceleration
        is sent. Messages to PX4 have timestamp 0: PX4 stamps them on arrival, so its clock offset can not lose them"""
        sp = self.setpoint
        if sp is None:
            return
        lp = self.latest["local_position"]
        self.smooth.update(sp["velocity"], sp["yaw"], 1.0 / SETPOINT_RATE)
        stale = time.monotonic() - self.stamp["local_position"] > LPOS_TIMEOUT
        too_fast = math.sqrt(lp.vx ** 2 + lp.vy ** 2 + lp.vz ** 2) > SPEED_GUARD
        if (stale or too_fast) and not self.guarding:
            print("-- " + ("No recent local position" if stale else "Too fast") + ": zero velocity commanded")
        self.guarding = stale or too_fast
        if self.guarding:
            self.smooth.reset(self.smooth.yaw)      # zero now, and from zero again afterwards

        # Stopped: hold the position, a velocity of 0 alone slowly drifts. z only in the air: PX4 takes off as soon
        # as a position setpoint is above the drone.
        vel, target = self.smooth.vel, sp["velocity"]
        stopped_xy = (self.phase in ("takeoff", "hover", "cruise", "approach") and not target[:2].any()
                      and math.hypot(vel[0], vel[1]) < STOP_SPEED)
        stopped_z = self.phase in ("hover", "cruise", "approach") and not target[2] and abs(vel[2]) < STOP_SPEED
        self.lock_xy = (self.lock_xy or (lp.x, lp.y)) if stopped_xy else None
        self.lock_z = (lp.z if self.lock_z is None else self.lock_z) if stopped_z else None

        mode = OffboardControlMode()
        mode.position = True    # position, or velocity on the axes where the position is NaN
        self.offboard_pub.publish(mode)
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.position = [*(self.lock_xy or (nan, nan)), nan if self.lock_z is None else self.lock_z]
        msg.velocity = [*((nan, nan) if self.lock_xy else (float(vel[0]), float(vel[1]))),
                        nan if self.lock_z is not None else float(vel[2])]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = float(self.smooth.yaw)
        self.setpoint_pub.publish(msg)

    def log_row(self):
        """One row of flight.csv (timer, LOG_RATE)"""
        if self.logger is None or any(n not in self.latest for n in ("local_position", "global_position", "status")):
            return
        lp, gp, st = self.latest["local_position"], self.latest["global_position"], self.latest["status"]
        height, _, source = self.height()
        sp = self.setpoint
        vel_target = sp["velocity"] if sp is not None else [math.nan] * 3
        sensing = self.phase in ("cruise", "approach")
        goal_az = math.degrees(math.atan2(self.goal[1] - lp.y, self.goal[0] - lp.x))
        row = {
            "node": self.get_name(),                                      # flight.csv is the same name for every script
            "t": time.monotonic() - self.t0, "phase": self.phase, "holding": int(self.holding),
            "guarding": int(self.guarding),
            "armed": int(st.arming_state == VehicleStatus.ARMING_STATE_ARMED), "nav_state": st.nav_state,
            "north": lp.x, "east": lp.y, "alt": self.start_z - lp.z,      # alt: EKF, above the take-off point
            "agl": lp.dist_bottom if source == "sensor" else math.nan,    # distance sensor, above the ground
            "height": height, "height_src": source,                       # height used for FLIGHT_ALT
            "vn": lp.vx, "ve": lp.vy, "vd": lp.vz, "speed": math.hypot(lp.vx, lp.vy),
            "an": lp.ax, "ae": lp.ay, "ad": lp.az,
            "lat": gp.lat, "lon": gp.lon, "alt_amsl": gp.alt,
            "heading": math.degrees(lp.heading), "yaw_cmd": math.degrees(self.smooth.yaw),
            "yaw_target": math.degrees(sp["yaw"]) if sp is not None else math.nan,
            "vel_target_n": vel_target[0], "vel_target_e": vel_target[1], "vel_target_d": vel_target[2],
            "vel_cmd_n": self.smooth.vel[0], "vel_cmd_e": self.smooth.vel[1], "vel_cmd_d": self.smooth.vel[2],
            "acc_cmd": float(np.linalg.norm(self.smooth.acc)),
            "goal_n": self.goal[0], "goal_e": self.goal[1],
            "alt_target": self.start_z - self.goal[2] if USE_GOAL_ALT else FLIGHT_ALT,
            "dist_goal": math.hypot(self.goal[0] - lp.x, self.goal[1] - lp.y), "goal_az": goal_az,
            "nearest_ahead": self.nearest if sensing else math.nan,       # obstacle on the way
            "nearest_all": self.nearest_all if sensing else math.nan,     # anything in the camera image
            # Columns flight_logger.plot() expects from the planner of obstacle_avoidance_static_gps.py.
            # There is no planner here: mode "blocked" marks the holds in the plots.
            "vfh_mode": "blocked" if self.holding else "goal", "dir_az": goal_az,
            "speed_factor": float(not self.holding) if sensing else math.nan,
            **{f"cost_{k}": math.nan for k in ("goal", "heading", "elev", "prev", "side", "total")},
        }
        self.logger.log(row)

    def fly_velocity(self, vel, yaw):
        """Every velocity target goes through here: horizontal and vertical together stay below MAX_SPEED"""
        self.setpoint = {"velocity": clip_norm(np.asarray(vel, dtype=float), MAX_SPEED), "yaw": yaw}

    def send_command(self, command, p1=0.0, p2=0.0, p3=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1, msg.param2, msg.param3 = float(p1), float(p2), float(p3)
        msg.target_system = self.latest["status"].system_id   # PX4 ignores commands for another system_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    def arm(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def offboard_mode(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)        # custom mode, OFFBOARD

    def hold_mode(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)   # AUTO, LOITER (Hold)

    # ---------- obstacles ----------
    def depth_points(self, pos, ground_z):
        """Newest depth image -> obstacle points in local NED (without the ground at local z ground_z)"""
        msg, info = self.latest["depth"], self.latest["camera_info"]
        depth = depth_to_meters(msg)[::DEPTH_STRIDE, ::DEPTH_STRIDE]
        if self.rays is None or self.rays.shape[:2] != depth.shape:
            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
            v, u = np.mgrid[0:msg.height:DEPTH_STRIDE, 0:msg.width:DEPTH_STRIDE].astype(np.float64)
            # Ray per pixel for depth = 1 m. Camera optical frame (z fwd, x right, y down) -> body FRD
            self.rays = np.stack([np.ones_like(u), (u - cx) / fx, (v - cy) / fy], axis=-1)

        valid = np.isfinite(depth) & (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        p_body = self.rays[valid] * depth[valid][:, None] + CAM_OFFSET
        p_ned = p_body @ quat_to_rot(self.latest["attitude"].q).T + pos
        return p_ned[p_ned[:, 2] < ground_z - GROUND_IGNORE]   # drop ground points (z is down)

    def nearest_on_way(self, goal):
        """Distance (m) to the closest obstacle on the way to the goal, along the way; inf if there is none.
        The way is a corridor of CORRIDOR_RADIUS around the straight level line from the drone to the goal, up to
        HOLD_DIST ahead and ending CORRIDOR_RADIUS behind the goal. Only the newest image counts (no memory of
        points), so the way is clear as soon as an obstacle has left it."""
        pos, heading = self.position()
        points = self.depth_points(pos, self.height()[1])
        if self.logger is not None:
            self.logger.add_obstacles(points)
        self.nearest_all = float(np.linalg.norm(points - pos, axis=1).min()) if len(points) else math.inf

        dist = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
        # Close to the goal the direction to it jumps with every cm of position noise: use the locked heading there
        az = math.atan2(goal[1] - pos[1], goal[0] - pos[0]) if dist > YAW_LOCK_DIST else heading
        axis = np.array([math.cos(az), math.sin(az), 0.0])
        rel = points - pos
        along = rel @ axis
        off_axis = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        on_way = along[(along > 0.0) & (along < min(HOLD_DIST, dist + CORRIDOR_RADIUS)) & (off_axis < CORRIDOR_RADIUS)]
        return float(on_way.min()) if len(on_way) >= MIN_OBSTACLE_POINTS else math.inf

    def way_blocked(self, goal):
        """True while the drone must hold: no camera image, an obstacle on the way, or the way has not yet been
        clear for CLEAR_TIME since the last stop"""
        now = time.monotonic()
        if now - self.stamp["depth"] > DEPTH_TIMEOUT:
            self.nearest, reason = math.nan, "No depth image"
        else:
            self.nearest = self.nearest_on_way(goal)
            reason = f"Obstacle {self.nearest:.1f} m ahead" if math.isfinite(self.nearest) else None

        if reason is not None:
            if not self.holding:
                print(f"-- {reason}, holding position (Ctrl+C: stop here in PX4 Hold)")
                self.holding = True
            self.hold_reason, self.clear_since = reason, None
            return True
        if not self.holding:
            return False
        if self.clear_since is None:
            self.hold_reason, self.clear_since = "way clear, waiting", now
        if now - self.clear_since < CLEAR_TIME:
            return True
        print("-- Way clear, continuing")
        self.holding = False
        return False

    def cruise_step(self, goal):
        """One step of the flight to the goal. Returns the horizontal distance to the goal"""
        pos, heading = self.position()
        dist_xy = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
        az = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        yaw = az if dist_xy > YAW_LOCK_DIST else heading     # also while holding: keeps the camera on the way

        source = self.height()[2]
        if source != self.height_source:
            print("-- Height: " + ("distance sensor (above the ground)" if source == "sensor" else
                                   "EKF (above the take-off point), no valid distance sensor reading"))
            self.height_source = source

        if self.way_blocked(goal):
            self.fly_velocity(np.zeros(3), yaw)     # brakes through the smoother, not with a position step
            return dist_xy

        speed = MAX_SPEED * min(1.0, dist_xy / SLOW_DOWN_DIST)
        if dist_xy > YAW_LOCK_DIST:     # don't move into a direction the camera does not see yet
            err = abs(wrap_pi(az - heading))
            speed *= float(np.clip((MAX_HEADING_ERR - err) / (MAX_HEADING_ERR - FULL_SPEED_HEADING_ERR), 0.0, 1.0))
        down = float(np.clip(ALT_GAIN * (self.target_z(goal) - pos[2]), -MAX_CLIMB, MAX_CLIMB))
        self.fly_velocity(np.array([speed * math.cos(az), speed * math.sin(az), down]), yaw)
        return dist_xy

    def velocity_towards(self, target):
        """Horizontal velocity (north, east) to the target, slowing down to it"""
        return clip_norm(APPROACH_GAIN * (target[:2] - self.position()[0][:2]), APPROACH_SPEED)


def switch_to_hold(node):
    """Keep streaming a velocity of 0 while asking for Hold: if the stream stopped first, PX4 would trigger its
    offboard loss failsafe (COM_OBL_RC_ACT)"""
    node.fly_velocity(np.zeros(3), node.smooth.yaw)
    t0 = time.monotonic()
    last_send = 0.0
    while time.monotonic() - t0 < HOLD_TIMEOUT:
        if node.latest["status"].nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
            return True
        if time.monotonic() - last_send > 0.5:
            node.hold_mode()
            last_send = time.monotonic()
        time.sleep(0.05)
    return False


def check_flying(node):
    if not node.is_armed():
        raise RuntimeError("Drone disarmed")
    if not node.in_offboard():
        raise RuntimeError("Drone left OFFBOARD (pilot took over, or a PX4 failsafe)")


def final_approach(node, goal):
    """From ACCEPT_RADIUS to above the center of the goal, heading and height unchanged. An obstacle or a lost
    camera image makes the drone hold here too: it never lands because of them"""
    node.phase = "approach"
    deadline = time.monotonic() + APPROACH_TIMEOUT
    settled_since = None
    while time.monotonic() < deadline:
        check_flying(node)
        if node.way_blocked(goal):
            node.fly_velocity(np.zeros(3), node.smooth.yaw)
            settled_since = None
            deadline = time.monotonic() + APPROACH_TIMEOUT     # the time limit is for flying, not for holding
            time.sleep(1.0 / LOOP_RATE)
            continue
        pos, _ = node.position()
        lp = node.latest["local_position"]
        if math.hypot(goal[0] - pos[0], goal[1] - pos[1]) < LAND_RADIUS and math.hypot(lp.vx, lp.vy) < 0.1:
            settled_since = settled_since or time.monotonic()
            if time.monotonic() - settled_since > SETTLE_TIME:
                return
        else:
            settled_since = None
        north, east = node.velocity_towards(goal)
        down = float(np.clip(ALT_GAIN * (node.target_z(goal) - pos[2]), -MAX_CLIMB, MAX_CLIMB))
        node.fly_velocity(np.array([north, east, down]), node.smooth.yaw)
        time.sleep(1.0 / LOOP_RATE)
    print("Not settled over the center of the goal in time, landing anyway")


def fly(node, goal):
    _, yaw = node.position()

    # OFFBOARD first and only then arm, so the drone is never armed in another mode.
    # The setpoints are already being streamed, PX4 refuses OFFBOARD without them.
    node.phase = "arming"
    node.smooth.reset(yaw)
    node.fly_velocity(np.zeros(3), yaw)
    time.sleep(1.0)
    t0 = time.monotonic()
    last_send = 0.0
    while not (node.is_armed() and node.in_offboard()):
        if time.monotonic() - t0 > 5.0:
            print("Could not arm / switch to OFFBOARD, check QGC messages")
            node.setpoint = None
            return
        if time.monotonic() - last_send > 1.0:    # first command may be lost, send again every second
            if not node.in_offboard():
                node.offboard_mode()
            elif not node.is_armed():
                node.arm()
            last_send = time.monotonic()
        time.sleep(0.1)
    print("-- Armed, OFFBOARD")

    # Take off straight up: climb at TAKEOFF_SPEED, brake gently (MAX_ACC / 4) before the flight
    # altitude, done when there and no longer climbing (the smoothed setpoint lags the target)
    node.phase = "takeoff"
    while True:
        check_flying(node)
        below = node.position()[0][2] - node.target_z(goal)     # NED: > 0 while below the flight altitude
        if abs(below) < 0.2 and abs(node.smooth.vel[2]) < 0.15:
            break
        climb = math.copysign(min(TAKEOFF_SPEED, math.sqrt(0.5 * MAX_ACC * abs(below))), below)
        node.fly_velocity(np.array([0.0, 0.0, -climb]), yaw)
        time.sleep(1.0 / LOOP_RATE)
    # Hover until the climb has completely stopped: braking vertically and accelerating forward
    # at the same time makes the drone lurch
    node.phase = "hover"
    node.fly_velocity(np.zeros(3), yaw)
    t_hover = time.monotonic()
    while time.monotonic() - t_hover < HOVER_TIME or \
            (np.linalg.norm(node.smooth.vel) > 0.02 and time.monotonic() - t_hover < 5.0):
        check_flying(node)
        time.sleep(0.1)
    height, _, source = node.height()
    lp = node.latest["local_position"]
    print(f"-- Take-off done: height {height:.2f} m ({source}), EKF {node.start_z - lp.z:.2f} m above the take-off "
          f"point, distance sensor {f'{lp.dist_bottom:.2f} m' if source == 'sensor' else 'no valid reading'}")
    print("-- Flying to goal")

    # Straight to the goal, holding while something is in the way
    node.phase = "cruise"
    last_print = 0.0
    while True:
        check_flying(node)
        dist = node.cruise_step(goal)
        if dist < ACCEPT_RADIUS:
            break
        if time.monotonic() - last_print > 2.0:
            height, _, source = node.height()
            state = f", holding: {node.hold_reason}" if node.holding else ""
            print(f"Distance to goal: {dist:.1f} m, height {height:.2f} m ({source}){state}")
            last_print = time.monotonic()
        time.sleep(1.0 / LOOP_RATE)
    print("-- Final approach to the center of the goal")
    final_approach(node, goal)

    # Land: down at LAND_SPEED until PX4 detects the landing and disarms
    node.phase = "landing"
    print("-- Landing")
    while node.is_armed():
        if not node.in_offboard():
            raise RuntimeError("Drone left OFFBOARD (pilot took over, or a PX4 failsafe)")
        north, east = node.velocity_towards(goal) if node.height()[0] > LAND_FIX_HEIGHT else (0.0, 0.0)
        node.fly_velocity(np.array([north, east, LAND_SPEED]), node.smooth.yaw)
        time.sleep(0.1)
    node.setpoint = None
    pos, _ = node.position()
    print(f"-- Landed and disarmed, {math.hypot(goal[0] - pos[0], goal[1] - pos[1]):.2f} m from the center of the goal")


def main():
    goal_lat, goal_lon, goal_alt = load_gps_point(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Goal: lat {goal_lat}, lon {goal_lon}, alt {goal_alt:.1f}")

    # Ctrl+C raises KeyboardInterrupt but keeps ROS2 running, so we can still send Hold
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ObstacleHoldGPS()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        print("Waiting for GPS and camera...")
        while not node.data_ready():
            time.sleep(0.5)
        print("-- Drone data OK")

        # Goal in the local NED frame (origin = PX4 local position reference)
        lp = node.latest["local_position"]
        node.start_z = lp.z
        north, east = gps_to_ned(goal_lat, goal_lon, lp.ref_lat, lp.ref_lon)
        goal_z = -(goal_alt - lp.ref_alt) if USE_GOAL_ALT else lp.z - FLIGHT_ALT
        goal_z = min(goal_z, lp.z - MIN_ALT)     # never lower than MIN_ALT above take-off
        goal = np.array([north, east, goal_z])
        node.goal = goal
        dist = math.hypot(north - lp.x, east - lp.y)
        print(f"Goal in local NED: ({north:.1f}, {east:.1f}, {goal_z:.1f}), {dist:.1f} m away")

        if input("Enter to take off, q to quit: ") != "q":
            node.logger = FlightLogger(LOG_DIR)
            print(f"Logging the flight to {node.logger.dir}")
            fly(node, goal)
    except RuntimeError as e:
        print(f"Stopped: {e}")
        node.setpoint = None
    except KeyboardInterrupt:
        print("Interrupted, switching to Hold")
        if "status" in node.latest and node.is_armed() and node.in_offboard():
            print("-- In Hold" if switch_to_hold(node) else "PX4 did not stay in Hold, check QGC messages")
        node.setpoint = None
    finally:
        # Also after an error / Ctrl+C: the log is what tells what went wrong
        logger, node.logger = node.logger, None
        if logger is not None:
            print("Saving the flight log and plots...")
            print(f"Flight log saved to {logger.save()}")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)   # Python aborts at exit if this thread is still inside ROS
        node.destroy_node()


if __name__ == "__main__":
    main()
