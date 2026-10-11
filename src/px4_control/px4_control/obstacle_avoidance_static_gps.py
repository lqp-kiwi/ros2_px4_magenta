#!/usr/bin/env python3

import sys
import json
import math
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, CameraInfo
from px4_msgs.msg import (VehicleGlobalPosition, VehicleLocalPosition, VehicleAttitude, VehicleStatus,
                          OffboardControlMode, TrajectorySetpoint, VehicleCommand)

from px4_control.vfh3d import VFH3D

"""
Script Name: go_to_gps_point.py
Description: Takes off, flies to the GPS point saved in a .json file by record_drone_state.py
             and lands there. Obstacles seen by the depth camera are avoided with 3DVFH+ (vfh3d.py).
             The camera node (depth_camera package) must already be running.

Last Modified: 2026-10-10

Usage:
    ros2 run px4_control obstacle_avoidance_static_gps [filename]
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
FLIGHT_ALT = 3.0          # flight altitude above the take-off point (m)
USE_GOAL_ALT = False      # True: fly at the altitude saved in the .json instead of FLIGHT_ALT
MIN_ALT = 1.5             # altitude band (above take-off point) allowed while avoiding (m)
MAX_ALT = 8.0
ACCEPT_RADIUS = 1.5       # goal reached when closer than this horizontally (m)
MAX_SPEED = 1.0           # (m/s)
SLOW_DOWN_DIST = 5.0      # start slowing down this far from the goal (m)
YAW_LOCK_DIST = 2.0       # closer than this to the goal: stop turning towards the flight direction (m)
MAX_HEADING_ERR = math.radians(35)  # only move when the camera looks within this angle of the flight direction
YAW_SCAN_RATE = math.radians(20)    # turn speed when every direction is blocked (rad/s)
PLAN_RATE = 10.0          # 3DVFH+ steps per second
SETPOINT_RATE = 20.0      # offboard setpoints per second (PX4 needs > 2 Hz)

# Depth camera
DEPTH_STRIDE = 8          # use every 8th pixel
DEPTH_MIN = 0.3           # (m)
DEPTH_MAX = 8.0           # (m)
DEPTH_TIMEOUT = 0.5       # hold position if no depth image for this long (s)
CAM_OFFSET = np.array([0.10, 0.0, 0.0])  # camera position in body FRD (m); camera faces forward, level
GROUND_IGNORE = 0.5       # ignore points lower than this above the take-off ground (m)

EARTH_RADIUS = 6371000.0

filename = ""


def filename_valid(filename: str) -> bool:
    dot_index = filename.find(".")
    return dot_index == -1 or filename[dot_index:] == ".json"


def load_goal(filename):
    """Read the (only) recorded state and return its GPS point"""
    with open(filename) as f:
        states = json.load(f)
    if len(states) != 1:
        print(f"Warning: {len(states)} states in {filename}, using the first one")
    gp = states[0]["global_position"]
    return gp["lat"], gp["lon"], gp["alt"]


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


class GoToGPSPoint(Node):
    def __init__(self):
        super().__init__("go_to_gps_point")
        self.latest = {}    # newest message of each topic
        self.stamp = {}     # time each topic was last received
        # PX4 publishes best-effort, so the subscriber QoS must match
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        for name, (topic, msg_type) in TOPICS.items():
            self.create_subscription(msg_type, topic, lambda msg, name=name: self.save_msg(name, msg), qos)
        # PX4 v1.16+ names it vehicle_status_v1, older versions vehicle_status
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1",
                                 lambda msg: self.save_msg("status", msg), qos)
        self.create_subscription(Image, DEPTH_TOPIC, lambda msg: self.save_msg("depth", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, lambda msg: self.save_msg("camera_info", msg),
                                 qos_profile_sensor_data)

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

        # Current setpoint, streamed by the timer (None = stream nothing)
        self.setpoint = None
        self.create_timer(1.0 / SETPOINT_RATE, self.publish_setpoint)

        self.vfh = VFH3D()
        self.start_z = 0.0      # local z of the take-off point
        self.prev_dir = None    # last direction flown (3DVFH+ smoothing)
        self.hold_pos = None    # set while holding because no path / no camera
        self.hold_yaw = 0.0
        self.rays = None        # pixel rays (body FRD) for the depth image

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

    # ---------- commands ----------
    def timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def publish_setpoint(self):
        sp = self.setpoint
        if sp is None:
            return
        mode = OffboardControlMode()
        mode.timestamp = self.timestamp()
        mode.position = sp["position"] is not None
        mode.velocity = sp["position"] is None
        self.offboard_pub.publish(mode)

        nan3 = [math.nan] * 3
        msg = TrajectorySetpoint()
        msg.timestamp = mode.timestamp
        msg.position = [float(v) for v in sp["position"]] if sp["position"] is not None else nan3
        msg.velocity = [float(v) for v in sp["velocity"]] if sp["velocity"] is not None else nan3
        msg.acceleration = nan3
        msg.jerk = nan3
        msg.yaw = float(sp["yaw"])
        msg.yawspeed = math.nan
        self.setpoint_pub.publish(msg)

    def hold_at(self, pos, yaw):
        self.setpoint = {"position": pos, "velocity": None, "yaw": yaw}

    def fly_velocity(self, vel, yaw):
        self.setpoint = {"position": None, "velocity": vel, "yaw": yaw}

    def send_command(self, command, p1=0.0, p2=0.0, p3=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp()
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

    def engage_offboard(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)        # custom mode, OFFBOARD

    def hold_mode(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)   # AUTO, LOITER (Hold)

    def land(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # ---------- obstacle avoidance ----------
    def depth_points(self, pos):
        """Newest depth image -> obstacle points in local NED"""
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
        return p_ned[p_ned[:, 2] < self.start_z - GROUND_IGNORE]   # drop ground points (z is down)

    def hold_here(self, pos, heading, reason, scan=False):
        if self.hold_pos is None:
            print(f"-- {reason}, holding position")
            self.hold_pos, self.hold_yaw = pos, heading
        if scan:    # turn slowly so the camera can look for a free direction
            self.hold_yaw = wrap_pi(self.hold_yaw + YAW_SCAN_RATE / PLAN_RATE)
        self.hold_at(self.hold_pos, self.hold_yaw)

    def plan_step(self, goal):
        """One 3DVFH+ step. Returns the horizontal distance to the goal"""
        pos, heading = self.position()
        dist_xy = math.hypot(goal[0] - pos[0], goal[1] - pos[1])

        if time.monotonic() - self.stamp["depth"] > DEPTH_TIMEOUT:
            self.hold_here(pos, heading, "No depth image")
            return dist_xy

        self.vfh.add_points(self.depth_points(pos), time.monotonic(), pos)
        alt = self.start_z - pos[2]
        direction = self.vfh.choose_direction(pos, goal, self.prev_dir, heading,
                                              allow_up=alt < MAX_ALT, allow_down=alt > MIN_ALT)
        if direction is None:
            self.hold_here(pos, heading, "All directions blocked", scan=True)
            return dist_xy
        if self.hold_pos is not None:
            print("-- Path found, continuing")
            self.hold_pos = None

        # Speed: slow near the goal and near obstacles
        speed = MAX_SPEED * min(1.0, dist_xy / SLOW_DOWN_DIST) * self.vfh.speed_factor

        # Point the camera where we fly; don't move into directions the camera hasn't seen
        yaw = heading
        az = math.atan2(direction[1], direction[0])
        if dist_xy > YAW_LOCK_DIST and math.hypot(direction[0], direction[1]) > 0.5:
            yaw = az
            err = abs(wrap_pi(az - heading))
            speed = 0.0 if err > MAX_HEADING_ERR else speed * math.cos(err)

        self.fly_velocity(direction * speed, yaw)
        self.prev_dir = direction
        return dist_xy


def check_offboard(node):
    if not node.in_offboard():
        raise RuntimeError("Drone left offboard mode (pilot took over)")


def fly(node, goal):
    pos, yaw = node.position()

    # Stream setpoints first, PX4 refuses offboard without them
    node.hold_at(pos, yaw)
    time.sleep(1.0)
    t0 = time.monotonic()
    last_send = 0.0
    while not (node.is_armed() and node.in_offboard()):
        if time.monotonic() - t0 > 5.0:
            print("Could not arm / switch to offboard, check QGC messages")
            node.setpoint = None
            return
        if time.monotonic() - last_send > 1.0:    # first command may be lost, send again every second
            if not node.in_offboard():
                node.engage_offboard()
            if not node.is_armed():
                node.arm()
            last_send = time.monotonic()
        time.sleep(0.1)
    print("-- Armed, offboard")

    # Take off straight up
    takeoff = np.array([pos[0], pos[1], goal[2]])
    node.hold_at(takeoff, yaw)
    while abs(node.position()[0][2] - goal[2]) > 0.3:
        check_offboard(node)
        time.sleep(0.1)
    print("-- Take-off done, flying to goal")

    # Fly to the goal with 3DVFH+
    last_print = 0.0
    while True:
        check_offboard(node)
        dist = node.plan_step(goal)
        if dist < ACCEPT_RADIUS:
            break
        if time.monotonic() - last_print > 2.0:
            print(f"Distance to goal: {dist:.1f} m")
            last_print = time.monotonic()
        time.sleep(1.0 / PLAN_RATE)
    print("-- Goal reached")

    # Hover, then land
    pos, yaw = node.position()
    node.hold_at(pos, yaw)
    time.sleep(2.0)
    check_offboard(node)
    node.land()
    node.setpoint = None
    print("-- Landing")
    while node.is_armed():
        time.sleep(0.5)
    print("-- Landed and disarmed")


def main():
    global filename
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        while True:
            filename = input("Enter the name of the .json file with the goal point: ")
            if filename_valid(filename):
                break
            print("This filename is invalid, because it contains a period, without ending in '.json'.")
    if filename.find(".") == -1:
        filename = filename + ".json"
    goal_lat, goal_lon, goal_alt = load_goal(filename)
    print(f"Goal: lat {goal_lat}, lon {goal_lon}, alt {goal_alt:.1f}")

    # Ctrl+C raises KeyboardInterrupt but keeps ROS2 running, so we can still send Hold
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = GoToGPSPoint()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

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
        dist = math.hypot(north - lp.x, east - lp.y)
        print(f"Goal in local NED: ({north:.1f}, {east:.1f}, {goal_z:.1f}), {dist:.1f} m away")

        if input("Enter to take off, q to quit: ") != "q":
            fly(node, goal)
    except RuntimeError as e:
        print(f"Stopped: {e}")
        node.setpoint = None
    except KeyboardInterrupt:
        print("Interrupted, switching to Hold")
        if "status" in node.latest and node.in_offboard():
            node.hold_mode()
            time.sleep(0.5)
        node.setpoint = None
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()