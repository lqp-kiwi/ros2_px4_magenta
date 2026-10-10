#!/usr/bin/env python3

"""
    Takeoff -> Offboard hold -> dodge a ball thrown at the drone -> go back to the hold point.
    PX4 + ROS2, OUTDOOR with GPS.
    Obstacle data comes from depth_grid.py (topic /depth_grid, n x n distance in meters, n >= 3).
    Authors: Phuong Le (lephuo10@rowan.edu)
    Last Updated: 2026-10-09

    Run (3 terminals):
        ros2 run depth_camera camera --no-cloud
        ros2 run depth_camera depth_grid
        python3 takeoff_dodge_ball_gps.py

    GPS part: same as takeoff_avoid_gps.py
        - Waits for a good GPS fix before ARM, TAKEOFF uses altitude AMSL, hold point saved as (lat, lon).
        - GPS fix lost -> do not dodge, just hold.

    Ball detection (on every new /depth_grid):
        closing speed of a cell = (distance VEL_WINDOW s ago - distance now) / time - drone's own forward speed
        A cell is a ball if: closer than DETECT_DIST and closing faster than MIN_CLOSING.
        time to hit = distance / closing speed. Dodge if < TTC_DODGE, seen in 2 grids in a row.
        Things that do not fly toward the drone (walls, trees, people standing) are ignored.

    Dodge direction: away from where the ball is in the image (camera looks forward, image left = drone left).
        ball on the left -> go right, ball high -> go down, ball low-right -> go up-left, ...
        ball straight at the center -> go left or right (the side with more space).
        UP / DOWN only inside [MIN_ALT, MAX_ALT]. Never farther than MAX_OFFSET from the hold point.
        Yaw never changes.

    States:
        HOLD   -> stay at the hold point (TAKEOFF_ALT above the ground)
        DODGE  -> jump DODGE_DIST away, as fast as possible
        WAIT   -> stay there until no ball for WAIT_TIME seconds
        RETURN -> fly slowly (RETURN_SPEED) back to the hold point -> HOLD
        A new ball in WAIT / RETURN / HOLD -> DODGE again from where the drone is.

    Bench test (props OFF): before Offboard, a detected ball is only printed ("no dodge"),
    so you can check detection by throwing / moving a ball toward the camera.

    Stays in Offboard until Ctrl+C. Land first (QGC or PX4 console: commander land) before stopping the node.
"""

import math
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition,
                          VehicleStatus, VehicleGlobalPosition, SensorGps)
from std_msgs.msg import Float32MultiArray

TAKEOFF_ALT = 3.0     # meters above the ground (hold altitude)
MIN_ALT = 2.0         # meters above the ground, never dodge DOWN below this
MAX_ALT = 6.0         # meters above the ground, never dodge UP above this
DETECT_DIST = 4.0     # meters, only look at things closer than this
MIN_CLOSING = 1.5     # m/s, slower than this is not a thrown ball (depth noise, things standing still)
TTC_DODGE = 1.0       # seconds, dodge if the ball would hit within this time
VEL_WINDOW = 0.15     # seconds, closing speed is measured over this window (less noise than frame to frame)
CENTER_ZONE = 0.3     # ball this close to the image center -> dodge sideways to the freer side
DODGE_DIST = 1.5      # meters moved per dodge
DODGE_SPEED = 3.0     # m/s, velocity feedforward at the start of a dodge
WAIT_TIME = 1.5       # seconds without a ball before going back
RETURN_SPEED = 0.5    # m/s, going back to the hold point (slow and smooth)
MAX_OFFSET = 5.0      # meters, never move farther than this from the hold point (geofence)
GRID_TIMEOUT = 0.5    # seconds, /depth_grid older than this -> warning
MIN_SATS = 8          # GPS satellites needed before ARM
EARTH_R = 6378137.0   # meters
NAN3 = (math.nan, math.nan, math.nan)

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


def find_ball(grid, old_grid, dt, own_speed):
    """Find the cell flying toward the drone with the shortest time to hit.
    Returns None or (row, col, distance, time_to_hit). NaN / inf cells (no data) are ignored.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        cur = np.where(np.isfinite(grid), grid, np.nan)
        old = np.where(np.isfinite(old_grid), old_grid, np.nan)
        closing = (old - cur) / dt - own_speed             # m/s toward the drone, own motion removed
        ttc = np.where((cur < DETECT_DIST) & (closing > MIN_CLOSING), cur / closing, np.inf)
    row, col = np.unravel_index(np.argmin(ttc), ttc.shape)
    if ttc[row, col] > TTC_DODGE:
        return None
    return int(row), int(col), float(cur[row, col]), float(ttc[row, col])


def dodge_direction(grid, row, col, can_up, can_down):
    """Unit (right, up) in body frame: away from where the ball is in the image."""
    n_rows, n_cols = grid.shape
    right = 1.0 - (col + 0.5) / n_cols * 2                 # ball at the left edge -> +1 (go right)
    up = (row + 0.5) / n_rows * 2 - 1.0                    # ball at the top edge -> -1 (go down)
    if (up > 0 and not can_up) or (up < 0 and not can_down):
        up = 0.0
    if math.hypot(right, up) < CENTER_ZONE:                # ball straight at us -> side with more space
        g = np.where(np.isnan(grid), np.inf, grid)
        half = n_cols // 2
        right = 1.0 if g[:, :half].min() <= g[:, -half:].min() else -1.0
    norm = math.hypot(right, up)
    return right / norm, up / norm


class TakeoffDodgeBallNode(Node):
    def __init__(self):
        super().__init__('takeoff_dodge_ball_node')

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

        # Obstacle grid from depth_grid.py, plus the grids of the last VEL_WINDOW seconds
        self.grid = None
        self.grid_time = 0.0
        self.history = deque()
        self.ball_count = 0       # grids in a row with a ball
        self.last_ball = 0.0      # time of the last ball seen
        self.create_subscription(Float32MultiArray, '/depth_grid', self.grid_callback, 10)

        # Timer for commands (duration 1 second)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.seconds_passed = 0   # stays 0 until GPS is good
        self.wait_time = 0
        self.target_system = 10

        # Timer for Offboard stream (50 Hz, fast reaction)
        self.dt = 0.02
        self.offboard_timer = self.create_timer(self.dt, self.offboard_timer_callback)

        # Setpoint state (local NED x, y, z). None = not streaming yet
        self.ground_z = None      # local z on the ground (saved at takeoff)
        self.home_xy = None       # hold point after takeoff (geofence center)
        self.home_gps = None      # hold point as (lat, lon)
        self.sp = None            # setpoint sent to PX4 right now
        self.target = None        # where the setpoint is moving to
        self.hold_yaw = None      # heading, never changes
        self.state = 'HOLD'
        self.dodge_start = 0.0
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

    def forward_speed(self):
        """Drone speed along the camera direction (m/s), so its own motion is not seen as a ball."""
        if self.local_pos is None:
            return 0.0
        yaw = self.hold_yaw if self.hold_yaw is not None else self.local_pos.heading
        return self.local_pos.vx * math.cos(yaw) + self.local_pos.vy * math.sin(yaw)

    def grid_callback(self, msg):
        n = msg.layout.dim[1].size if len(msg.layout.dim) >= 2 else int(round(math.sqrt(len(msg.data))))
        self.grid = np.array(msg.data, dtype=float).reshape(-1, n)
        self.grid_time = time.monotonic()

        # Keep the oldest grid that is still about VEL_WINDOW old, used for the closing speed
        self.history.append((self.grid_time, self.grid))
        while len(self.history) > 1 and self.history[1][0] <= self.grid_time - VEL_WINDOW:
            self.history.popleft()
        old_time, old_grid = self.history[0]
        if old_grid.shape != self.grid.shape or self.grid_time - old_time < VEL_WINDOW / 2:
            return

        ball = find_ball(self.grid, old_grid, self.grid_time - old_time, self.forward_speed())
        self.ball_count = self.ball_count + 1 if ball else 0
        if ball is None or self.ball_count < 2:
            return
        if not (self.avoid_enabled and self.is_offboard() and self.gps_ok()):
            self.get_logger().info(f">>> BALL {ball[2]:.2f} m, hits in {ball[3]:.2f} s "
                                   f"(not flying in Offboard - no dodge) <<<", throttle_duration_sec=0.5)
            return
        self.last_ball = self.grid_time
        if self.state != 'DODGE':
            self.start_dodge(*ball)

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

        # Turn dodging on once PX4 is really in Offboard. Stays on until Ctrl+C.
        if self.seconds_passed >= 15 and not self.avoid_enabled and self.is_offboard():
            self.avoid_enabled = True
            self.get_logger().info(">>> In OFFBOARD - ball dodging ON (Ctrl+C to stop) <<<")
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
            st = f"sys_id={self.status.system_id} {armed} mode={mode} state={self.state}"
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
        if self.avoid_enabled and not self.is_offboard():
            self.get_logger().warn(">>> Not in OFFBOARD (pilot or failsafe took over) - no dodge <<<",
                                   throttle_duration_sec=5.0)
        elif self.avoid_enabled and time.monotonic() - self.grid_time > GRID_TIMEOUT:
            self.get_logger().warn(">>> No fresh /depth_grid - cannot see balls. Is depth_grid running? <<<",
                                   throttle_duration_sec=2.0)

        vel = NAN3
        if self.state == 'DODGE':
            vel = self.dodge_velocity()
        elif self.state == 'WAIT' and time.monotonic() - self.last_ball > WAIT_TIME:
            self.state = 'RETURN'
            self.target = (self.home_xy[0], self.home_xy[1], self.ground_z - TAKEOFF_ALT)
            self.get_logger().info(">>> No ball - going back to the hold point <<<")
        elif self.state == 'RETURN':
            self.move_setpoint_toward_target()
            if self.sp == self.target:
                self.state = 'HOLD'
                self.get_logger().info(">>> Back at the hold point <<<")

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(self.sp[0], self.sp[1], self.sp[2], self.hold_yaw, vel)

    def start_dodge(self, row, col, dist, ttc):
        lp = self.local_pos
        alt = self.ground_z - lp.z                         # altitude above the ground
        right, up = dodge_direction(self.grid, row, col,
                                    can_up=alt + 0.5 <= MAX_ALT, can_down=alt - 0.5 >= MIN_ALT)

        # Body frame (right, up) -> local NED (north, east, down) using the hold heading, from where we are now
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        x = lp.x - right * DODGE_DIST * s
        y = lp.y + right * DODGE_DIST * c
        new_alt = min(max(alt + up * DODGE_DIST, MIN_ALT), MAX_ALT)

        # Geofence: stay within MAX_OFFSET of the hold point
        dx, dy = x - self.home_xy[0], y - self.home_xy[1]
        d = math.hypot(dx, dy)
        if d > MAX_OFFSET:
            x, y = self.home_xy[0] + dx * MAX_OFFSET / d, self.home_xy[1] + dy * MAX_OFFSET / d

        # Jump the setpoint: PX4 flies there as fast as it can (plus velocity feedforward)
        self.target = (x, y, self.ground_z - new_alt)
        self.sp = self.target
        self.state = 'DODGE'
        self.dodge_start = time.monotonic()
        lat, lon = offset_to_gps(self.home_gps, x - self.home_xy[0], y - self.home_xy[1])
        self.get_logger().info(
            f">>> BALL {dist:.2f} m, hits in {ttc:.2f} s -> dodge right={right:+.2f} up={up:+.2f} "
            f"(target lat={lat:.7f} lon={lon:.7f} alt={new_alt:.1f} m) <<<")
        self.offboard_timer_callback()                     # send now, do not wait for the next tick

    def dodge_velocity(self):
        """Velocity feedforward toward the dodge target: fast at the start, slows down near the target."""
        lp = self.local_pos
        d = (self.target[0] - lp.x, self.target[1] - lp.y, self.target[2] - lp.z)
        dist = math.hypot(*d)
        if dist < 0.3 or time.monotonic() - self.dodge_start > 3.0:
            self.state = 'WAIT'
            return NAN3
        speed = min(DODGE_SPEED, 2.0 * dist)
        return tuple(v * speed / dist for v in d)

    def move_setpoint_toward_target(self):
        d = [t - p for t, p in zip(self.target, self.sp)]
        dist = math.hypot(*d)
        max_step = RETURN_SPEED * self.dt
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
        self.state = 'HOLD'
        self.get_logger().info(
            f">>> Start streaming hold setpoint lat={self.home_gps[0]:.7f} lon={self.home_gps[1]:.7f} "
            f"alt={TAKEOFF_ALT}m <<<")

    def set_offboard_mode(self):
        # param1 = 1 (custom mode), param2 = 6 (PX4 OFFBOARD mode)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(f">>> Sent OFFBOARD mode to Vehicle {self.target_system} <<<")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True     # control by position setpoint (velocity is only feedforward)
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw, vel=NAN3):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float(v) for v in vel]
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
    node = TakeoffDodgeBallNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()