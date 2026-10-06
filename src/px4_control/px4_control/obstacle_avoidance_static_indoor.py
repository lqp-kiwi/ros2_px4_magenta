#!/usr/bin/env python3

"""
    Takeoff -> Offboard hold -> dodge obstacles (keep altitude) for PX4 using ROS2 in door.
    Obstacle data comes from depth_grid.py (topic /depth_grid, n x n mean distance in meters).
    Author: Phuong Le
    Lasted Updated: 2026-10-06

    Run (3 terminals):
        ros2 run depth_camera camera --no-cloud
        ros2 run depth_camera depth_grid
        python3 takeoff_avoid_indoor.py

    Dodge rule (camera looks forward, image left = drone left):
        - Use the middle rows of the grid (skip top = ceiling, bottom = floor)
        - If any cell is closer than DANGER_DIST:
            obstacle more on the left  -> move right
            obstacle more on the right -> move left
            both sides blocked         -> move back
        - Altitude and yaw never change. Never go farther than MAX_OFFSET from the hold point.

    Stays in Offboard and keeps dodging forever, until you press Ctrl+C.
    Land first (QGC or PX4 console: commander land) before stopping the node.

    Status line (sys_id, armed, mode, local position valid) is printed every second for the first
    20 s, then every 5 s, so you can see at which step it stops.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition, VehicleStatus
from std_msgs.msg import Float32MultiArray

TAKEOFF_ALT = 1.5     # meters (indoor: keep it low)
DANGER_DIST = 1.0     # meters, obstacle closer than this -> dodge
DODGE_STEP = 0.5      # meters moved per dodge
MAX_SPEED = 0.3       # m/s, the setpoint moves at most this fast (smooth, no jumps)
MAX_OFFSET = 2.0      # meters, never move farther than this from the hold point (indoor geofence)
GRID_TIMEOUT = 0.5    # seconds, /depth_grid older than this -> do not dodge, just hold

# nav_state number -> name (e.g. 17 -> AUTO_TAKEOFF), read from the message definition
NAV_STATE_NAMES = {getattr(VehicleStatus, n): n.replace('NAVIGATION_STATE_', '')
                   for n in dir(VehicleStatus) if n.startswith('NAVIGATION_STATE_') and n != 'NAVIGATION_STATE_MAX'}


def choose_dodge(grid, danger_dist):
    """Decide where to dodge from the n x n grid.
    Returns (direction, closest) with direction in {None, 'left', 'right', 'back'}.
    NaN cells (no data) are ignored.
    """
    rows = grid[1:-1] if grid.shape[0] >= 3 else grid      # skip ceiling / floor rows
    rows = np.where(np.isnan(rows), np.inf, rows)
    col_min = rows.min(axis=0)                             # closest distance in each column
    closest = float(col_min.min())
    if closest >= danger_dist:
        return None, closest

    half = len(col_min) // 2
    left_free = float(col_min[:half].min())                # worst cell on the left half
    right_free = float(col_min[len(col_min) - half:].min())  # worst cell on the right half
    if max(left_free, right_free) < danger_dist:
        return 'back', closest
    return ('left' if left_free > right_free else 'right'), closest


class TakeoffAvoidNode(Node):
    def __init__(self):
        super().__init__('takeoff_avoid_node')

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

        # Local position (x, y, z in NED) - available indoor, no GPS needed
        self.local_pos = None
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.local_pos_callback, qos_profile)

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
        self.seconds_passed = 0
        self.target_system = 10

        # Timer for Offboard stream + avoidance (10 Hz)
        self.dt = 0.1
        self.offboard_timer = self.create_timer(self.dt, self.offboard_timer_callback)

        # Setpoint state (local NED). None = not streaming yet
        self.home_xy = None       # hold point after takeoff (geofence center)
        self.sp_xy = None         # setpoint sent to PX4 right now
        self.target_xy = None     # where the setpoint is moving to
        self.hold_z = None        # altitude, never changes
        self.hold_yaw = None      # heading, never changes
        self.avoid_enabled = False

    def local_pos_callback(self, msg):
        self.local_pos = msg

    def status_callback(self, msg):
        self.status = msg

    def is_armed(self):
        return self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def is_offboard(self):
        return self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def grid_callback(self, msg):
        n = msg.layout.dim[1].size if len(msg.layout.dim) >= 2 else int(round(math.sqrt(len(msg.data))))
        self.grid = np.array(msg.data, dtype=float).reshape(-1, n)
        self.grid_time = time.monotonic()

    def timer_callback(self):
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
            lp = (f"xy_valid={self.local_pos.xy_valid} z_valid={self.local_pos.z_valid} "
                  f"x={self.local_pos.x:.2f} y={self.local_pos.y:.2f} alt={-self.local_pos.z:.2f} m")
        self.get_logger().info(f"[t={self.seconds_passed:3d}s] {st} | {lp}")

    def offboard_timer_callback(self):
        if self.sp_xy is None:
            return
        if self.avoid_enabled:
            self.avoid_step()
        self.move_setpoint_toward_target()
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(self.sp_xy[0], self.sp_xy[1], self.hold_z, self.hold_yaw)

    def avoid_step(self):
        # Pilot / QGC / failsafe switched away from Offboard -> do not move the setpoint
        if not self.is_offboard():
            self.get_logger().warn(">>> Not in OFFBOARD (pilot or failsafe took over) - avoidance paused <<<",
                                   throttle_duration_sec=5.0)
            return
        # No fresh obstacle data -> do not move, just hold
        if self.grid is None or time.monotonic() - self.grid_time > GRID_TIMEOUT:
            self.get_logger().warn(">>> No fresh /depth_grid - holding. Is depth_grid running? <<<",
                                   throttle_duration_sec=2.0)
            return
        # Wait until the previous dodge is finished before choosing a new one
        if math.dist(self.sp_xy, self.target_xy) > 0.05:
            return

        direction, closest = choose_dodge(self.grid, DANGER_DIST)
        if direction is None:
            return

        # Body frame (forward, right) -> local NED (north, east) using the hold heading
        fwd, right = {'left': (0.0, -DODGE_STEP), 'right': (0.0, DODGE_STEP), 'back': (-DODGE_STEP, 0.0)}[direction]
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        new_x = self.target_xy[0] + fwd * c - right * s
        new_y = self.target_xy[1] + fwd * s + right * c

        # Indoor geofence: stay within MAX_OFFSET of the hold point
        dx, dy = new_x - self.home_xy[0], new_y - self.home_xy[1]
        dist = math.hypot(dx, dy)
        if dist > MAX_OFFSET:
            new_x = self.home_xy[0] + dx * MAX_OFFSET / dist
            new_y = self.home_xy[1] + dy * MAX_OFFSET / dist
        if math.dist((new_x, new_y), self.target_xy) < 0.05:
            self.get_logger().warn(f">>> Obstacle {closest:.2f} m but geofence limit reached - holding <<<",
                                   throttle_duration_sec=2.0)
            return

        self.target_xy = (new_x, new_y)
        self.get_logger().info(f">>> Obstacle {closest:.2f} m -> dodge {direction.upper()} <<<")

    def move_setpoint_toward_target(self):
        dx = self.target_xy[0] - self.sp_xy[0]
        dy = self.target_xy[1] - self.sp_xy[1]
        dist = math.hypot(dx, dy)
        max_step = MAX_SPEED * self.dt
        if dist <= max_step:
            self.sp_xy = self.target_xy
        else:
            self.sp_xy = (self.sp_xy[0] + dx * max_step / dist, self.sp_xy[1] + dy * max_step / dist)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f">>> Sent ARM to Vehicle {self.target_system} <<<")

    def takeoff(self, altitude=1.5):
        # Param7 in NAV_TAKEOFF: altitude to take off (meters)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=float(altitude))
        self.get_logger().info(f">>> Sent TAKEOFF command (altitude: {altitude}m) to Vehicle {self.target_system} <<<")

    def start_hold(self):
        if self.local_pos is None:
            self.get_logger().error(">>> No vehicle_local_position received! Check topic name <<<")
            return
        # Hold at current x, y, heading. z = -TAKEOFF_ALT (NED: negative = up)
        self.home_xy = (self.local_pos.x, self.local_pos.y)
        self.sp_xy = self.home_xy
        self.target_xy = self.home_xy
        self.hold_z = -TAKEOFF_ALT
        self.hold_yaw = self.local_pos.heading
        self.get_logger().info(
            f">>> Start streaming hold setpoint x={self.local_pos.x:.2f} y={self.local_pos.y:.2f} alt={TAKEOFF_ALT}m <<<")

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

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.target_system = self.target_system
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TakeoffAvoidNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()