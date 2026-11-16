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

from px4_control.vfh3d import VFH3D
from px4_control.flight_logger import FlightLogger
from px4_control.gps_point import load_gps_point

"""
Script Name: obstacle_avoidance_static_offboard.py
Description: Takes off, flies to the GPS point saved in a .json file by record_drone_state.py
             and lands there. Obstacles seen by the depth camera are avoided with 3DVFH+ (vfh3d.py).
             The camera node (depth_camera package) must already be running.
             Copy of obstacle_avoidance_static_gps.py (2026-11-16), same OFFBOARD flight code.
             The whole flight is in OFFBOARD: from before arming until PX4 disarms after landing, the node streams
             velocity setpoints over uXRCE-DDS. The link of MicroXRCEAgent is the only one it needs, so the same code
             flies in SITL and on the real drone; only the agent is started differently (Usage).
             Stopped, the drone holds its position, as in Position mode with centered sticks.
             Take-off: a quick lift-off (LIFTOFF_SPEED), then a slow climb (TAKEOFF_SPEED).
             Sitting on the ground at nearly hover thrust makes the drone rock and yaw at lift-off.
             No local position from PX4 for LPOS_TIMEOUT: the drone brakes smoothly and waits.
             A radio keeps working: switching PX4 to another mode ends the node, the drone is then the pilot's.
             Position mode is not possible here: PX4 1.14 does not bridge /fmu/in/manual_control_input over
             uXRCE-DDS, virtual sticks would need a second link (MAVLink, the version before 2026-11-15).

PX4 parameters:
    COM_OF_LOSS_T        OFFBOARD is lost this long after the last setpoint (default 1 s), e.g. when the node dies.
    COM_OBL_RC_ACT       what PX4 then does. Ctrl+C does not trigger it: the node switches to Hold before it ends.
    COM_RC_OVERRIDE      bit 1 (value 3): moving the sticks of the radio also takes over from OFFBOARD. Default 1:
                         only the mode switch of the radio does.
    SER_*_BAUD           of the uXRCE-DDS port: PX4 1.14 sends about 78 KB/s (measured in SITL),
                         115200 baud carries 11.5 KB/s: most messages are dropped, the stream
                         can stall. Use 921600 or more.
    MPC_THR_HOVER        the real hover thrust (hover_thrust_estimate of a flight log), else
                         the lift-off is slow.

Last Modified: 2026-11-16

Usage:
    MicroXRCEAgent udp4 -p 8888                              SITL
    MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600       real drone (Orin), same baud as the SER_*_BAUD of
                                                             the flight controller port in UXRCE_DDS_CFG
    ros2 run px4_control obstacle_avoidance_static_offboard           point of record_drone_state
    ros2 run px4_control obstacle_avoidance_static_offboard filename  first point of this .json
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

# Flight
FLIGHT_ALT = 2.0          # flight height above the ground below the drone, measured by the downward distance
                          # sensor; above the take-off point (EKF) while the sensor has no valid reading (m)
USE_GOAL_ALT = False      # True: fly at the altitude saved in the .json instead of FLIGHT_ALT
MIN_ALT = 1             # height band (same reference as FLIGHT_ALT) allowed while avoiding (m)
MAX_ALT = 8.0
ALT_GAIN = 0.5            # vertical speed per meter of height error when holding FLIGHT_ALT (1/s)
MAX_CLIMB = 0.3           # max vertical speed when holding FLIGHT_ALT (m/s)
ACCEPT_RADIUS = 1.5       # closer than this horizontally: 3DVFH+ ends, straight to the center of the goal (m)
LAND_RADIUS = 0.15        # start landing when closer than this to the center and slow (m)
APPROACH_SPEED = 0.3      # max horizontal speed to the center (m/s)
APPROACH_GAIN = 0.4       # horizontal speed per meter from the center (1/s)
APPROACH_TIMEOUT = 60.0   # land anyway when not settled over the center after this long (s)
SETTLE_TIME = 1.0         # stay within LAND_RADIUS this long before landing (s)
LAND_FIX_HEIGHT = 0.5     # no more horizontal corrections below this height, for a straight touchdown (m)
MAX_SPEED = 0.5           # horizontal speed in cruise, lower near obstacles (m/s)
SLOW_DOWN_DIST = 1.5      # start slowing down this far from the goal (m)
YAW_LOCK_DIST = 2.0       # closer than this to the goal: stop turning towards the flight direction (m)
MAX_HEADING_ERR = math.radians(50)  # only move when the camera looks within this angle of where the drone goes
FULL_SPEED_HEADING_ERR = math.radians(25)  # full speed below this heading error, slower up to MAX_HEADING_ERR
YAW_SCAN_RATE = math.radians(20)    # turn speed when every direction is blocked (rad/s)
PLAN_RATE = 10.0          # 3DVFH+ steps per second
SETPOINT_RATE = 20.0      # OFFBOARD setpoints per second (PX4 leaves OFFBOARD without them, COM_OF_LOSS_T)
# Telemetry: over the 921600 baud link of the real drone PX4's local position stalled for 3.5 s
# (obstacle_hold_gps, flight 20261109_153650). Meanwhile the node does not know where the drone is.
LPOS_TIMEOUT = 0.5        # no local position for this long: brake smoothly to a stop and wait (s)
STOP_SPEED = 0.02         # below this and not asked to move: the position is held instead of a velocity of 0 (m/s)
TAKEOFF_SPEED = 0.35      # climb speed during take-off, once off the ground (m/s)
# Lift-off: with a slow climb PX4 raises the thrust only through its integrator (MPC_THR_HOVER
# below the real hover thrust). Seconds on the ground at nearly hover thrust wind up the attitude
# integrators, and the drone rocks and yaws when it lifts off. So it climbs faster until it is up.
# Off the ground = climbing at half of LIFTOFF_SPEED, or the distance sensor LIFTOFF_HEIGHT higher.
# Not the EKF height: near the ground the baro drifts (+0.26 m on the ground, 0.8 m off at 0.9 m).
LIFTOFF_SPEED = 1.0       # climb speed asked for until the drone has left the ground (m/s)
LIFTOFF_HEIGHT = 0.3      # (m)
LAND_SPEED = 0.35         # descent speed when landing (m/s)
LAND_SPEED_MIN = 0.35     # near the ground: above 1.1 * LNDMC_Z_VEL_MAX, or PX4 never detects the landing (m/s)
LAND_GAIN = 0.5           # descent speed per meter of height when landing (1/s)
HOLD_TIMEOUT = 10.0       # time to get PX4 to stay in Hold after Ctrl+C (s)
HOVER_TIME = 1.0          # hover after take-off before flying forward, so the climb has fully stopped (s)
SPEED_HOLD_TIME = 1.0     # an obstacle ahead slows the drone down for at least this long (no speed flicker) (s)

# Setpoint smoothing: a step in velocity or yaw makes the drone jerk (shakes the camera, can crash a
# real drone). The velocity setpoint is jerk-limited, the yaw setpoint acceleration-limited. Low
# values = smooth flight (tilt ~ MAX_ACC / g), but wider turns around obstacles.
MAX_ACC = 0.5            # (m/s^2)
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
# The real camera sees the front propellers (nearest returns 0.3-0.53 m). Obstacles seen
# from farther away stay in the 3DVFH+ point memory (memory_time) as the drone gets closer.
DEPTH_MIN = 1.0           # ignore depth closer than this (m)
DEPTH_MAX = 12.0          # (m), >= VFH3D hist_radius so obstacles are seen early enough
DEPTH_TIMEOUT = 0.5       # hold position if no depth image for this long (s)
CAM_OFFSET = np.array([0.10, 0.0, 0.0])  # camera position in body FRD (m); camera faces forward, level
GROUND_IGNORE = 0.5       # ignore points lower than this above the ground (below the drone, else take-off point) (m)

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
    whatever the planner asks for, the setpoints sent to PX4 never jump"""

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


class GoToGPSPoint(Node):
    def __init__(self):
        super().__init__("obstacle_avoidance_static_offboard")
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
        self.stale = False      # True while the local position is older than LPOS_TIMEOUT
        self.create_timer(1.0 / SETPOINT_RATE, self.publish_setpoints)

        self.vfh = VFH3D()
        self.start_z = 0.0      # local z of the take-off point
        self.goal = None        # goal in local NED
        self.prev_dir = None    # last direction flown (3DVFH+ smoothing)
        self.holding = False    # True while stopped because no path / no camera
        self.hold_yaw = 0.0
        self.speed_factors = []     # (time, 3DVFH+ speed factor) of the last SPEED_HOLD_TIME
        self.height_source = None   # "sensor" or "ekf", to print when it changes
        self.rays = None        # pixel rays (body FRD) for the depth image

        # Flight log
        self.logger = None      # FlightLogger while flying
        self.phase = "wait"     # wait, arming, takeoff, hover, cruise, approach, landing
        self.plan_ms = math.nan     # duration of the last planning step
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

    def lpos_stale(self):
        return time.monotonic() - self.stamp["local_position"] > LPOS_TIMEOUT

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
    # Messages to PX4 have timestamp 0: PX4 stamps them on arrival. With the ROS clock PX4 converts the time with
    # its estimate of the clock offset, which drifts by seconds when SITL runs slower than real time: PX4 then
    # treats the setpoints as old and refuses or leaves OFFBOARD.
    def publish_setpoints(self):
        """OFFBOARD stream (timer): the smoothed velocity and yaw, or the position where the drone stopped (a
        velocity of 0 alone slowly drifts). No acceleration is sent, the yaw rate goes with the yaw as feedforward"""
        sp = self.setpoint
        if sp is None:
            return
        lp = self.latest["local_position"]
        # No recent local position: brake through the smoother (no setpoint step, unlike a
        # reset) and keep the heading. PX4 itself still flies on its own estimate.
        stale = self.lpos_stale()
        if stale != self.stale:
            print(f"-- No local position for {LPOS_TIMEOUT} s: braking and waiting" if stale
                  else "-- Local position from PX4 again")
            self.stale = stale
        target = np.zeros(3) if stale else sp["velocity"]
        self.smooth.update(target, self.smooth.yaw if stale else sp["yaw"], 1.0 / SETPOINT_RATE)

        # Stopped: hold the position, taken from fresh data only. z only in the air: PX4 takes off
        # as soon as a position setpoint is above the drone. Not on the ground before lift-off
        # (phase "liftoff"): the EKF drifts there, the drone would dart to the old point.
        vel = self.smooth.vel
        stopped_xy = (self.phase in ("takeoff", "hover", "cruise", "approach") and not target[:2].any()
                      and math.hypot(vel[0], vel[1]) < STOP_SPEED)
        stopped_z = self.phase in ("hover", "cruise", "approach") and not target[2] and abs(vel[2]) < STOP_SPEED
        if not stopped_xy:
            self.lock_xy = None
        elif self.lock_xy is None and not stale:
            self.lock_xy = (lp.x, lp.y)
        if not stopped_z:
            self.lock_z = None
        elif self.lock_z is None and not stale:
            self.lock_z = lp.z

        mode = OffboardControlMode()
        mode.timestamp = 0
        mode.position = True    # position, or velocity on the axes where the position is NaN
        self.offboard_pub.publish(mode)
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.timestamp = 0
        msg.position = [*(self.lock_xy or (nan, nan)), nan if self.lock_z is None else self.lock_z]
        msg.velocity = [*((nan, nan) if self.lock_xy else (float(vel[0]), float(vel[1]))),
                        nan if self.lock_z is not None else float(vel[2])]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = float(self.smooth.yaw)
        msg.yawspeed = float(self.smooth.yaw_rate)
        self.setpoint_pub.publish(msg)

    def log_row(self):
        """One row of flight.csv (timer, LOG_RATE)"""
        if self.logger is None or any(n not in self.latest for n in ("local_position", "global_position", "status")):
            return
        lp, gp, st = self.latest["local_position"], self.latest["global_position"], self.latest["status"]
        height, _, source = self.height()
        sp = self.setpoint
        vel_target = sp["velocity"] if sp is not None else [math.nan] * 3
        planning = self.phase == "cruise"
        sensing = planning or self.phase == "approach"
        d = self.prev_dir if planning and self.prev_dir is not None else [math.nan] * 3
        terms = self.vfh.cost_terms if planning else {}
        row = {
            "t": time.monotonic() - self.t0, "phase": self.phase, "holding": int(self.holding),
            "stale": int(self.stale),                                     # old local position
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
            "dist_goal": math.hypot(self.goal[0] - lp.x, self.goal[1] - lp.y),
            "goal_az": math.degrees(math.atan2(self.goal[1] - lp.y, self.goal[0] - lp.x)),
            "vfh_mode": self.vfh.mode if planning else "", "vfh_side": self.vfh.side if planning else 0,
            "dir_az": math.degrees(math.atan2(d[1], d[0])),
            "dir_el": math.degrees(math.asin(max(-1.0, min(1.0, -d[2])))) if planning and self.prev_dir is not None
            else math.nan,
            **{f"cost_{k}": terms.get(k, math.nan) for k in ("goal", "heading", "elev", "prev", "side")},
            "cost_total": sum(terms.values()) if terms else math.nan,
            "nearest_ahead": self.vfh.nearest_ahead if sensing else math.nan,
            "nearest_all": float(self.vfh.raw.min()) if sensing else math.nan,
            "speed_factor": self.vfh.speed_factor if planning else math.nan,
            "obstacle_points": len(self.vfh.mem_pts), "plan_ms": self.plan_ms,
        }
        self.logger.log(row)

    def fly_velocity(self, vel, yaw):
        self.setpoint = {"velocity": np.asarray(vel, dtype=float), "yaw": yaw}

    def send_command(self, command, p1=0.0, p2=0.0, p3=0.0):
        msg = VehicleCommand()
        msg.timestamp = 0
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

    # ---------- obstacle avoidance ----------
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

    def hold_here(self, heading, reason, scan=False):
        """Brake smoothly (velocity 0 through the smoother, not a position step) and wait"""
        if not self.holding:
            print(f"-- {reason}, holding position")
            self.holding, self.hold_yaw = True, heading
        if scan:    # turn slowly so the camera can look for a free direction
            self.hold_yaw = wrap_pi(self.hold_yaw + YAW_SCAN_RATE / PLAN_RATE)
        self.fly_velocity(np.zeros(3), self.hold_yaw)

    def plan_step(self, goal):
        """One 3DVFH+ step. Returns the horizontal distance to the goal"""
        t_start = time.perf_counter()
        pos, heading = self.position()
        dist_xy = math.hypot(goal[0] - pos[0], goal[1] - pos[1])

        if self.lpos_stale():   # old pose: depth points would go to wrong places in memory
            self.hold_here(heading, "No local position from PX4")
            return dist_xy
        if time.monotonic() - self.stamp["depth"] > DEPTH_TIMEOUT:
            self.hold_here(heading, "No depth image")
            return dist_xy

        # Goal at the flight altitude above the ground below the drone (distance sensor), else EKF
        height, ground_z, source = self.height()
        if source != self.height_source:
            print("-- Height: " + ("distance sensor (above the ground)" if source == "sensor" else
                                   "EKF (above the take-off point), no valid distance sensor reading"))
            self.height_source = source
        goal_here = np.array([goal[0], goal[1], goal[2] if USE_GOAL_ALT else ground_z - FLIGHT_ALT])

        points = self.depth_points(pos, ground_z)
        self.vfh.add_points(points, time.monotonic(), pos)
        direction = self.vfh.choose_direction(pos, goal_here, self.prev_dir, heading,
                                              allow_up=height < MAX_ALT, allow_down=height > MIN_ALT)
        self.plan_ms = (time.perf_counter() - t_start) * 1000
        to_goal = goal_here - pos
        goal_dir = to_goal / max(float(np.linalg.norm(to_goal)), 1e-6)
        if self.logger is not None:
            self.logger.add_obstacles(points)
            self.logger.add_step(time.monotonic() - self.t0, (pos[0], pos[1], self.start_z - pos[2]), self.vfh,
                                 goal_dir, direction)
        if direction is None:
            self.hold_here(heading, "All directions blocked", scan=True)
            return dist_xy
        if self.holding:
            print("-- Path found, continuing")
            self.holding = False

        # Speed: slow near the goal and near obstacles. The obstacle factor is the lowest of the last
        # SPEED_HOLD_TIME, so points flickering in and out of view give one slow-down, not several.
        now = time.monotonic()
        self.speed_factors = [(t, f) for t, f in self.speed_factors if now - t < SPEED_HOLD_TIME]
        self.speed_factors.append((now, self.vfh.speed_factor))
        speed = MAX_SPEED * min(1.0, dist_xy / SLOW_DOWN_DIST) * min(f for _, f in self.speed_factors)

        # Point the camera where we fly; don't move into directions the camera hasn't seen
        yaw = heading
        az = math.atan2(direction[1], direction[0])
        if dist_xy > YAW_LOCK_DIST and math.hypot(direction[0], direction[1]) > 0.5:
            yaw = az
            # Slow down gradually (not a sudden stop) when the camera does not look where the drone is
            # really going: the smoothed velocity, which turns towards a new direction together with
            # the camera (a new 3DVFH+ direction alone must not brake the drone)
            vel = self.smooth.vel
            move_az = math.atan2(vel[1], vel[0]) if math.hypot(vel[0], vel[1]) > 0.3 else az
            err = abs(wrap_pi(move_az - heading))
            speed *= float(np.clip((MAX_HEADING_ERR - err) / (MAX_HEADING_ERR - FULL_SPEED_HEADING_ERR), 0.0, 1.0))

        vel = direction * speed
        # Hold the flight altitude with its own controller: the goal direction alone climbs or sinks far
        # too slowly while the goal is far away. Not while 3DVFH+ climbs on purpose over an obstacle
        # (then its direction is steeper than the goal direction).
        goal_el = math.asin(float(np.clip(-goal_dir[2], -1.0, 1.0)))
        if abs(math.asin(float(np.clip(-direction[2], -1.0, 1.0))) - goal_el) < math.radians(1):
            vel[2] = float(np.clip(ALT_GAIN * (goal_here[2] - pos[2]), -MAX_CLIMB, MAX_CLIMB))
        self.fly_velocity(vel, yaw)
        self.prev_dir = direction
        return dist_xy

    def velocity_towards(self, target):
        """Horizontal velocity (north, east) to the target, slowing down to it"""
        return clip_norm(APPROACH_GAIN * (target[:2] - self.position()[0][:2]), APPROACH_SPEED)

    def way_clear(self, target):
        """Final approach: False if the camera shows an obstacle closer than the safety radius on the way
        to the target, or there is no camera image"""
        if time.monotonic() - self.stamp["depth"] > DEPTH_TIMEOUT:
            return False
        pos, _ = self.position()
        points = self.depth_points(pos, self.height()[1])
        self.vfh.add_points(points, time.monotonic(), pos)
        self.vfh.build_histogram(pos)
        if self.logger is not None:
            self.logger.add_obstacles(points)
        to_target = target[:2] - pos[:2]
        dist = float(np.linalg.norm(to_target))
        if dist < 0.05:     # there: no direction to look at
            return True
        self.vfh.update_speed_factor(np.array([to_target[0] / dist, to_target[1] / dist, 0.0]))
        return self.vfh.nearest_ahead > self.vfh.safety_radius


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
    """From ACCEPT_RADIUS to above the center of the goal, heading and height unchanged.
    False if an obstacle or a lost camera image keeps the drone from going on."""
    node.phase = "approach"
    t0 = time.monotonic()
    settled_since = None
    while time.monotonic() - t0 < APPROACH_TIMEOUT:
        check_flying(node)
        if node.lpos_stale():   # wait for it (publish_setpoints brakes), no landing because of it
            settled_since = None
            time.sleep(1.0 / PLAN_RATE)
            continue
        if not node.way_clear(goal):
            return False
        pos, _ = node.position()
        lp = node.latest["local_position"]
        if math.hypot(goal[0] - pos[0], goal[1] - pos[1]) < LAND_RADIUS and math.hypot(lp.vx, lp.vy) < 0.1:
            settled_since = settled_since or time.monotonic()
            if time.monotonic() - settled_since > SETTLE_TIME:
                return True
        else:
            settled_since = None
        north, east = node.velocity_towards(goal)
        down = float(np.clip(ALT_GAIN * (node.target_z(goal) - pos[2]), -MAX_CLIMB, MAX_CLIMB))
        node.fly_velocity(np.array([north, east, down]), node.smooth.yaw)
        time.sleep(1.0 / PLAN_RATE)
    print("Not settled over the center of the goal in time, landing anyway")
    return True


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

    # Lift off quickly (LIFTOFF_SPEED), no hold position yet (see publish_setpoints)
    node.phase = "liftoff"
    t_start = time.monotonic()
    h_start, _, src_start = node.height()
    floor = (h_start if src_start == "sensor" else 0.0) + LIFTOFF_HEIGHT
    while True:
        check_flying(node)
        h, _, src = node.height()
        climbing = node.latest["local_position"].vz < -0.5 * LIFTOFF_SPEED
        if climbing or (src == "sensor" and h > floor):
            break
        node.fly_velocity(np.array([0.0, 0.0, -LIFTOFF_SPEED]), yaw)
        time.sleep(1.0 / PLAN_RATE)
    print(f"-- Off the ground after {time.monotonic() - t_start:.1f} s")

    # Climb at TAKEOFF_SPEED holding the point where it lifted off, brake gently (MAX_ACC / 4)
    # before the flight altitude, done when there and no longer climbing (the setpoint lags)
    node.phase = "takeoff"
    while True:
        check_flying(node)
        below = node.position()[0][2] - node.target_z(goal)     # NED: > 0 while below the flight altitude
        if abs(below) < 0.2 and abs(node.smooth.vel[2]) < 0.15:
            break
        climb = math.copysign(min(TAKEOFF_SPEED, math.sqrt(0.5 * MAX_ACC * abs(below))), below)
        node.fly_velocity(np.array([0.0, 0.0, -climb]), yaw)
        time.sleep(1.0 / PLAN_RATE)
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

    # Fly to the goal with 3DVFH+
    node.phase = "cruise"
    last_print = 0.0
    while True:
        check_flying(node)
        dist = node.plan_step(goal)
        if dist < ACCEPT_RADIUS:
            break
        if time.monotonic() - last_print > 2.0:
            height, _, source = node.height()
            print(f"Distance to goal: {dist:.1f} m, height {height:.2f} m ({source})")
            last_print = time.monotonic()
        time.sleep(1.0 / PLAN_RATE)
    print("-- Final approach to the center of the goal")

    target = goal
    if not final_approach(node, goal):
        print("-- Obstacle close ahead or no camera image: landing here, not over the center of the goal")
        target = node.position()[0]

    # Land: descend until PX4 detects the landing and disarms
    node.phase = "landing"
    print("-- Landing")
    while node.is_armed():
        if not node.in_offboard():
            raise RuntimeError("Drone left OFFBOARD (pilot took over, or a PX4 failsafe)")
        height = node.height()[0]
        descent = float(np.clip(LAND_GAIN * height, LAND_SPEED_MIN, LAND_SPEED))
        north, east = node.velocity_towards(target) if height > LAND_FIX_HEIGHT else (0.0, 0.0)
        node.fly_velocity(np.array([north, east, descent]), node.smooth.yaw)
        time.sleep(0.1)
    node.setpoint = None
    time.sleep(3.0 / LOG_RATE)  # rows with the drone disarmed: flight_eval.py reads the last row
    pos, _ = node.position()
    print(f"-- Landed and disarmed, {math.hypot(goal[0] - pos[0], goal[1] - pos[1]):.2f} m from the center of the goal")


def main():
    goal_lat, goal_lon, goal_alt = load_gps_point(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Goal: lat {goal_lat}, lon {goal_lon}, alt {goal_alt:.1f}")

    # Ctrl+C raises KeyboardInterrupt but keeps ROS2 running, so we can still send Hold
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = GoToGPSPoint()
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