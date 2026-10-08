#!/usr/bin/env python3

"""
    Takeoff -> Offboard hold -> dodge obstacles in 7 directions for PX4 using ROS2, OUTDOOR with GPS.
    Obstacle data comes from depth_grid.py (topic /depth_grid, n x n mean distance in meters, n >= 3).
    Authors: Phuong Le (lephuo10@rowan.edu)
    Lasted Updated: 2026-10-08

    Run (3 terminals):
        ros2 run depth_camera camera --no-cloud
        ros2 run depth_camera depth_grid
        python3 takeoff_avoid_gps.py

    GPS part:
        - Waits for a good GPS fix (3D fix, >= MIN_SATS satellites, local position valid) before ARM.
        - TAKEOFF uses GPS altitude: param7 = ground altitude AMSL + TAKEOFF_ALT,
          lat/lon = NaN (= take off right here).
        - Hold point (home) is saved in GPS (lat, lon). Status line shows lat, lon and distance to home.
        - Each dodge prints the target as GPS (lat, lon) and altitude above the ground.
        - GPS fix lost -> do not dodge, just hold.
        Setpoints are still sent in local NED: PX4 EKF already fuses GPS into local position.

    Dodge rule (camera looks forward, image left = drone left):
        The grid is cut into bands:
                 +--------+----------+---------+
                 |            top              |   -> is it free to go UP?
                 +--------+----------+---------+
                 |  left  |  center  |  right  |   -> where is the obstacle?
                 +--------+----------+---------+
                 |           bottom            |   -> is it free to go DOWN?
                 +--------+----------+---------+
        A band is "blocked" if any cell is closer than DANGER_DIST.
            only left blocked                 -> RIGHT
            only right blocked                -> LEFT
            left + right blocked (center free)-> BACK
            center blocked, both sides free   -> LEFT or RIGHT (the side with more space)
            center + right blocked            -> BACK_LEFT
            center + left blocked             -> BACK_RIGHT
            left + center + right blocked     -> UP if top free, else DOWN if bottom free, else BACK
        UP / DOWN only inside [MIN_ALT, MAX_ALT] above the ground. Yaw never changes.
        Never go farther than MAX_OFFSET (horizontal) from the hold point.

    Stays in Offboard and keeps dodging forever, until you press Ctrl+C.
    Land first (QGC or PX4 console: commander land) before stopping the node.

    Status line (sys_id, armed, mode, local position, GPS) is printed every second for the first
    20 s, then every 5 s, so you can see at which step it stops.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition,
                          VehicleStatus, VehicleGlobalPosition, SensorGps)
from std_msgs.msg import Float32MultiArray

TAKEOFF_ALT = 3.0     # meters above the ground
MIN_ALT = 2.0         # meters above the ground, never dodge DOWN below this
MAX_ALT = 6.0         # meters above the ground, never dodge UP above this
DANGER_DIST = 1.5     # meters, obstacle closer than this -> dodge
DODGE_STEP = 1.0      # meters moved per dodge
MAX_SPEED = 0.5       # m/s, the setpoint moves at most this fast (smooth, no jumps)
MAX_OFFSET = 5.0      # meters, never move farther than this from the hold point (geofence)
GRID_TIMEOUT = 0.5    # seconds, /depth_grid older than this -> do not dodge, just hold
MIN_SATS = 8          # GPS satellites needed before ARM
EARTH_R = 6378137.0   # meters

# Dodge directions in body frame: (forward, right, up), length 1
S = math.sqrt(0.5)
DIRECTIONS = {
    'left':       (0.0, -1.0, 0.0),
    'right':      (0.0, 1.0, 0.0),
    'back':       (-1.0, 0.0, 0.0),
    'back_left':  (-S, -S, 0.0),
    'back_right': (-S, S, 0.0),
    'up':         (0.0, 0.0, 1.0),
    'down':       (0.0, 0.0, -1.0),
}

# nav_state number -> name (e.g. 17 -> AUTO_TAKEOFF), read from the message definition
NAV_STATE_NAMES = {getattr(VehicleStatus, n): n.replace('NAVIGATION_STATE_', '')
                   for n in dir(VehicleStatus) if n.startswith('NAVIGATION_STATE_') and n != 'NAVIGATION_STATE_MAX'}


def gps_dist(a, b):
    """Distance in meters between two (lat, lon) points. Flat earth, fine for < 1 km."""
    dn = math.radians(b[0] - a[0]) * EARTH_R
    de = math.radians(b[1] - a[1]) * EARTH_R * math.cos(math.radians(a[0]))
    return math.hypot(dn, de)


def offset_to_gps(home, north, east):
    """(lat, lon) of a point north/east meters away from home (lat, lon)."""
    lat = home[0] + math.degrees(north / EARTH_R)
    lon = home[1] + math.degrees(east / (EARTH_R * math.cos(math.radians(home[0]))))
    return lat, lon


def choose_dodge(grid, danger_dist, can_up=True, can_down=True):
    """Decide where to dodge from the n x n grid (see the table at the top of the file).
    Returns (direction, closest) with direction = None or a key of DIRECTIONS.
    NaN cells (no data, e.g. sky) are ignored.
    """
    g = np.where(np.isnan(grid), np.inf, grid)
    if min(g.shape) < 3:                                   # too small to cut into bands
        closest = float(g.min())
        return ('back' if closest < danger_dist else None), closest

    k, m = g.shape[0] // 3, g.shape[1] // 3                # band size (rows, columns)
    top, bottom = float(g[:k].min()), float(g[-k:].min())
    mid = g[k:-k]                                          # middle rows
    left, center, right = float(mid[:, :m].min()), float(mid[:, m:-m].min()), float(mid[:, -m:].min())

    closest = min(left, center, right)
    if closest >= danger_dist:
        return None, closest
    free_l, free_c, free_r = left >= danger_dist, center >= danger_dist, right >= danger_dist

    if free_c:                                             # obstacle only at the side(s)
        if free_l:
            return 'left', closest
        if free_r:
            return 'right', closest
        return 'back', closest
    # obstacle in front
    if free_l and free_r:
        return ('left' if left > right else 'right'), closest
    if free_l:
        return 'back_left', closest
    if free_r:
        return 'back_right', closest
    if can_up and top >= danger_dist:
        return 'up', closest
    if can_down and bottom >= danger_dist:
        return 'down', closest
    return 'back', closest


class TakeoffAvoidGpsNode(Node):
    def __init__(self):
        super().__init__('takeoff_avoid_gps_node')

        # Config QoS for PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        # Local position (x, y, z in NED) - with GPS fused by the EKF
        self.local_pos = None
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.local_pos_callback, qos_profile)

        # GPS: global position (lat, lon, alt AMSL) and raw receiver (fix type, satellites)
        self.global_pos = None
        self.gps = None
        self.create_subscription(
            VehicleGlobalPosition, '/fmu/out/vehicle_global_position',
            self.global_pos_callback, qos_profile)
        self.create_subscription(
            SensorGps, '/fmu/out/vehicle_gps_position', self.gps_callback, qos_profile)

        # Vehicle status (armed, mode, system_id).
        # PX4 v1.16+ names it vehicle_status_v1, older versions vehicle_status
        self.status = None
        for topic in ('/fmu/out/vehicle_status', '/fmu/out/vehicle_status_v1'):
            self.create_subscription(VehicleStatus, topic, self.status_callback, qos_profile)

        # Obstacle grid from depth_grid.py
        self.grid = None
        self.grid_time = 0.0
        self.create_subscription(Float32MultiArray, '/depth_grid', self.grid_callback, 10)

        # Timer for commands (duration 1 second) - same as before
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.seconds_passed = 0   # stays 0 until GPS is good
        self.wait_time = 0
        self.target_system = 10

        # Timer for Offboard stream + avoidance (10 Hz)
        self.dt = 0.1
        self.offboard_timer = self.create_timer(self.dt, self.offboard_timer_callback)

        # Setpoint state (local NED x, y, z). None = not streaming yet
        self.ground_z = None      # local z on the ground (saved at takeoff)
        self.home_xy = None       # hold point after takeoff (geofence center)
        self.home_gps = None      # hold point as (lat, lon)
        self.sp = None            # setpoint sent to PX4 right now
        self.target = None        # where the setpoint is moving to
        self.hold_yaw = None      # heading, never changes
        self.avoid_enabled = False

    def local_pos_callback(self, msg):
        self.local_pos = msg

    def global_pos_callback(self, msg):
        self.global_pos = msg

    def gps_callback(self, msg):
        self.gps = msg

    def status_callback(self, msg):
        self.status = msg

    def is_armed(self):
        return self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def is_offboard(self):
        return self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def gps_ok(self):
        return (self.gps is not None and self.gps.fix_type >= 3 and self.gps.satellites_used >= MIN_SATS
                and self.global_pos is not None and self.local_pos is not None and self.local_pos.xy_valid)

    def grid_callback(self, msg):
        n = msg.layout.dim[1].size if len(msg.layout.dim) >= 2 else int(round(math.sqrt(len(msg.data))))
        self.grid = np.array(msg.data, dtype=float).reshape(-1, n)
        self.grid_time = time.monotonic()

    def timer_callback(self):
        # Wait for a good GPS fix before starting the sequence
        if self.seconds_passed == 0 and not self.gps_ok():
            self.wait_time += 1
            if self.wait_time % 2 == 1:
                self.print_status()
                self.get_logger().warn(f">>> Waiting for GPS (3D fix, >= {MIN_SATS} satellites) before ARM <<<")
            return

        self.seconds_passed += 1
        if self.seconds_passed <= 20 or self.seconds_passed % 5 == 0:
            self.print_status()

        if self.seconds_passed == 1:
            self.check_target_system()
            self.arm()
        elif self.seconds_passed == 3 and self.status is not None and not self.is_armed():
            # First ARM may be lost if the DDS link was not ready yet -> send again
            self.get_logger().warn(">>> Not armed after 2 s, sending ARM again <<<")
            self.arm()
        elif self.seconds_passed == 5:
            self.takeoff(altitude=TAKEOFF_ALT)
        elif self.seconds_passed == 13:
            # Takeoff should be done by now: start streaming the hold setpoint
            self.start_hold()
        elif self.seconds_passed in (15, 17, 19) and not self.avoid_enabled:
            # After 2 seconds of streaming, switch to Offboard (retry twice if PX4 did not switch)
            self.set_offboard_mode()

        # Turn avoidance on once PX4 is really in Offboard. Stays on until Ctrl+C.
        if self.seconds_passed >= 15 and not self.avoid_enabled and self.is_offboard():
            self.avoid_enabled = True
            self.get_logger().info(">>> In OFFBOARD - obstacle avoidance ON (Ctrl+C to stop) <<<")
        elif self.seconds_passed == 20 and not self.avoid_enabled:
            self.get_logger().error(">>> PX4 did not enter OFFBOARD - check the status lines above <<<")

    def check_target_system(self):
        if self.status is None:
            self.get_logger().error(
                ">>> No vehicle_status from PX4! Is MicroXRCEAgent running? Check: ros2 topic list | grep fmu <<<")
            return
        if self.status.system_id != self.target_system:
            self.get_logger().error(
                f">>> PX4 system_id = {self.status.system_id} but target_system = {self.target_system}. "
                f"PX4 ignores these commands! Using {self.status.system_id} instead <<<")
            self.target_system = self.status.system_id

    def print_status(self):
        if self.status is None:
            st = "status: (none)"
        else:
            armed = "ARMED" if self.is_armed() else "DISARMED"
            mode = NAV_STATE_NAMES.get(self.status.nav_state, str(self.status.nav_state))
            st = f"sys_id={self.status.system_id} {armed} mode={mode}"
        if self.local_pos is None:
            lp = "local_pos: (none)"
        else:
            lp = (f"xy_valid={self.local_pos.xy_valid} "
                  f"x={self.local_pos.x:.2f} y={self.local_pos.y:.2f} z={self.local_pos.z:.2f}")
        if self.gps is None:
            gp = "gps: (none) - check vehicle_gps_position in dds_topics.yaml"
        else:
            gp = f"fix={self.gps.fix_type} sats={self.gps.satellites_used}"
            if self.global_pos is not None:
                here = (self.global_pos.lat, self.global_pos.lon)
                gp += f" lat={here[0]:.7f} lon={here[1]:.7f}"
                if self.home_gps is not None:
                    gp += f" home_dist={gps_dist(self.home_gps, here):.1f} m"
        self.get_logger().info(f"[t={self.seconds_passed:3d}s] {st} | {lp} | {gp}")

    def offboard_timer_callback(self):
        if self.sp is None:
            return
        if self.avoid_enabled:
            self.avoid_step()
        self.move_setpoint_toward_target()
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(self.sp[0], self.sp[1], self.sp[2], self.hold_yaw)

    def avoid_step(self):
        # Pilot / QGC / failsafe switched away from Offboard -> do not move the setpoint
        if not self.is_offboard():
            self.get_logger().warn(">>> Not in OFFBOARD (pilot or failsafe took over) - avoidance paused <<<",
                                   throttle_duration_sec=5.0)
            return
        # GPS lost -> position may drift, do not dodge
        if not self.gps_ok():
            self.get_logger().warn(">>> GPS fix lost - holding (no dodge) <<<", throttle_duration_sec=2.0)
            return
        # No fresh obstacle data -> do not move, just hold
        if self.grid is None or time.monotonic() - self.grid_time > GRID_TIMEOUT:
            self.get_logger().warn(">>> No fresh /depth_grid - holding. Is depth_grid running? <<<",
                                   throttle_duration_sec=2.0)
            return
        # Wait until the previous dodge is finished before choosing a new one
        if math.dist(self.sp, self.target) > 0.05:
            return

        alt = self.ground_z - self.target[2]               # altitude above the ground
        direction, closest = choose_dodge(self.grid, DANGER_DIST,
                                          can_up=alt + DODGE_STEP <= MAX_ALT + 1e-3,
                                          can_down=alt - DODGE_STEP >= MIN_ALT - 1e-3)
        if direction is None:
            return

        # Body frame (forward, right, up) -> local NED (north, east, down) using the hold heading
        fwd, right, up = (DODGE_STEP * v for v in DIRECTIONS[direction])
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        new_x = self.target[0] + fwd * c - right * s
        new_y = self.target[1] + fwd * s + right * c
        new_z = self.target[2] - up                        # NED: up = negative z

        # Geofence: stay within MAX_OFFSET of the hold point
        dx, dy = new_x - self.home_xy[0], new_y - self.home_xy[1]
        dist = math.hypot(dx, dy)
        if dist > MAX_OFFSET:
            new_x = self.home_xy[0] + dx * MAX_OFFSET / dist
            new_y = self.home_xy[1] + dy * MAX_OFFSET / dist
        if math.dist((new_x, new_y, new_z), self.target) < 0.05:
            self.get_logger().warn(f">>> Obstacle {closest:.2f} m but geofence limit reached - holding <<<",
                                   throttle_duration_sec=2.0)
            return

        self.target = (new_x, new_y, new_z)
        lat, lon = offset_to_gps(self.home_gps, new_x - self.home_xy[0], new_y - self.home_xy[1])
        self.get_logger().info(
            f">>> Obstacle {closest:.2f} m -> dodge {direction.upper()} "
            f"(target lat={lat:.7f} lon={lon:.7f} alt={self.ground_z - new_z:.1f} m) <<<")

    def move_setpoint_toward_target(self):
        d = [t - p for t, p in zip(self.target, self.sp)]
        dist = math.hypot(*d)
        max_step = MAX_SPEED * self.dt
        if dist <= max_step:
            self.sp = self.target
        else:
            self.sp = tuple(p + v * max_step / dist for p, v in zip(self.sp, d))

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f">>> Sent ARM to Vehicle {self.target_system} <<<")

    def takeoff(self, altitude=3.0):
        if self.global_pos is None or self.local_pos is None:
            self.get_logger().error(">>> No GPS / local position - TAKEOFF not sent <<<")
            return
        # With GPS, param7 is altitude AMSL (above sea level) -> ground altitude + altitude.
        # param5/param6 (lat/lon) = NaN -> take off right here (0.0 would mean lat 0, lon 0!)
        amsl = self.global_pos.alt + altitude
        self.ground_z = self.local_pos.z
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                                     param4=math.nan, param5=math.nan, param6=math.nan, param7=float(amsl))
        self.get_logger().info(
            f">>> Sent TAKEOFF command ({altitude}m above ground = {amsl:.1f}m AMSL) to Vehicle {self.target_system} <<<")

    def start_hold(self):
        if self.local_pos is None or self.global_pos is None:
            self.get_logger().error(">>> No vehicle_local_position / vehicle_global_position received! <<<")
            return
        if self.ground_z is None:
            self.ground_z = 0.0
        # Hold at current x, y, heading, TAKEOFF_ALT above the ground (NED: negative = up)
        self.home_xy = (self.local_pos.x, self.local_pos.y)
        self.home_gps = (self.global_pos.lat, self.global_pos.lon)
        self.sp = (self.local_pos.x, self.local_pos.y, self.ground_z - TAKEOFF_ALT)
        self.target = self.sp
        self.hold_yaw = self.local_pos.heading
        self.get_logger().info(
            f">>> Start streaming hold setpoint lat={self.home_gps[0]:.7f} lon={self.home_gps[1]:.7f} "
            f"alt={TAKEOFF_ALT}m <<<")

    def set_offboard_mode(self):
        # param1 = 1 (custom mode), param2 = 6 (PX4 OFFBOARD mode)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(f">>> Sent OFFBOARD mode to Vehicle {self.target_system} <<<")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True     # control by position setpoint
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param4=0.0, param5=0.0, param6=0.0,
                                param7=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.target_system = self.target_system
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TakeoffAvoidGpsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()