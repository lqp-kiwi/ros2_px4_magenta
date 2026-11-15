#!/usr/bin/env python3
"""
    Takeoff and Land for PX4 using ROS2 with GPS, flown in Position (GPS) mode instead of OFFBOARD.
    After the PX4 takeoff (MIS_TAKEOFF_ALT) the drone switches to Position mode and the node moves a
    virtual throttle stick (/fmu/in/manual_control_input) until the drone is at the height given as
    argument above the ground right below it, measured by the downward TeraRanger
    (vehicle_local_position.dist_bottom), or above the arming point (EKF) while the sensor has no valid
    reading. Roll / pitch / yaw sticks stay centered, so PX4 holds the GPS position and heading.
    It hovers there for HOVER_TIME, then lands.
    Ctrl+C / closing the terminal while in Position mode switches PX4 to Hold before the node exits.

    PX4 parameters:
        COM_RC_IN_MODE = 1   PX4 takes the node's stick (MAVLink only, SITL default). With a radio and
                             3 (keep first) the radio is used and the node's stick is ignored.
        COM_RCL_EXCEPT = 2   once PX4 has used the node's stick, losing it (node stopped mid-flight)
                             triggers the manual control loss failsafe NAV_RCL_ACT (Return);
                             2 = ignore it in Auto modes (Hold, Takeoff, Land)
    Author: Phuong Le
    Lasted Updated: 2026-10-26
    GPS and height are logged to flight_logs/<date>_<time>/<node>.csv while flying.
    Usage:
	    ros2 run px4_control takeoff_land_mission_gps [height_m]     (default HOLD_HEIGHT)
"""

import csv
import math
import os
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from px4_msgs.msg import (
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
    FailsafeFlags,
    SensorGps,
    ManualControlSetpoint,
)

NAN = float('nan')

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TARGET_SYSTEM = 1         # must match MAV_SYS_ID on the flight controller
REQUIRE_GPS = True        # outdoor: wait for GPS/home like ensure_gps_estimate(); False indoors
HOVER_TIME = 10.0         # s to hover once the target height is reached
HOLD_HEIGHT = 3.0         # m, target height when no argument is given: above the ground below the
                          # drone (distance sensor), above the arming point (EKF) while it has no valid reading
HEIGHT_TOLERANCE = 0.15   # m, target height counts as reached within this error (stick centered)

# Throttle stick [-1, 1] while the height is off. PX4 Position mode: climb rate =
# MPC_Z_VEL_MAX_UP / _DN * expo(deadzone(stick, MAN_DEADZONE 0.1), 0.6)
STICK_GAIN = 0.3          # stick per m of height error
STICK_MIN = 0.2           # just outside the deadzone: ~0.14 m/s up, 0.07 m/s down (SITL defaults)
STICK_MAX = 0.4           # ~0.47 m/s up, 0.23 m/s down (SITL defaults)

READY_TIMEOUT = 120.0     # s to wait for PX4 to be ready
ARM_TIMEOUT = 10.0        # s to wait for ARMED
TAKEOFF_TIMEOUT = 30.0    # s to wait for takeoff to finish (then land)
CLIMB_TIMEOUT = 60.0      # s to reach the target height after takeoff (then land)
STOP_TIMEOUT = 3.0        # s to wait for PX4 to leave Position mode after Ctrl+C / terminal closed
RESEND_PERIOD = 1.0       # s between command retries
LOG_DIR = 'flight_logs'   # csv of GPS and height per flight, relative to the folder the node is started from

TOPIC_CMD = '/fmu/in/vehicle_command'
TOPIC_MANUAL = '/fmu/in/manual_control_input'
TOPIC_STATUS = '/fmu/out/vehicle_status'
TOPIC_LPOS = '/fmu/out/vehicle_local_position'
TOPIC_FAILSAFE = '/fmu/out/failsafe_flags'
TOPIC_GPS = '/fmu/out/vehicle_gps_position'


def field(msg, name, default=None):
    """Read a message field safely (field names can differ between PX4 versions)."""
    return getattr(msg, name, default) if msg is not None else default


def versioned(topic, msg_type):
    """PX4 v1.16+ adds _v<MESSAGE_VERSION> to the topic name of versioned messages
    (e.g. vehicle_status_v4). Older PX4 uses the plain name."""
    version = getattr(msg_type, 'MESSAGE_VERSION', 0)
    return f'{topic}_v{version}' if version else topic


class TakeoffLandMissionPosition(Node):
    def __init__(self, height):
        super().__init__('takeoff_land_mission_gps')
        self.height = height      # m, target height above the ground

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_pub = self.create_publisher(VehicleCommand, TOPIC_CMD, qos)
        # Publish on both the plain and the versioned name: works with old and new PX4
        self.manual_pubs = [self.create_publisher(ManualControlSetpoint, topic, qos)
                            for topic in {TOPIC_MANUAL, versioned(TOPIC_MANUAL, ManualControlSetpoint)}]

        # Listen to both the plain and the versioned name: works with old and new PX4
        for topic in {TOPIC_STATUS, versioned(TOPIC_STATUS, VehicleStatus)}:
            self.create_subscription(VehicleStatus, topic, self.status_cb, qos)
        for topic in {TOPIC_LPOS, versioned(TOPIC_LPOS, VehicleLocalPosition)}:
            self.create_subscription(VehicleLocalPosition, topic, self.lpos_cb, qos)
        self.create_subscription(FailsafeFlags, TOPIC_FAILSAFE, self.failsafe_cb, qos)
        self.create_subscription(SensorGps, TOPIC_GPS, self.gps_cb, qos)

        self.status = None
        self.lpos = None
        self.failsafe = None
        self.gps = None

        self.state = 'WAIT_READY'
        self.state_start = self.now()
        self.last_cmd_time = 0.0
        self.last_log_time = 0.0
        self.seen_takeoff = False
        self.z0 = 0.0
        self.interrupted = False  # set by Ctrl+C / terminal closed, handled in loop()
        self.log_file = None      # csv of GPS and height, opened once flying (log_flight)
        self.log_writer = None
        self.log_start = 0.0

        self.timer = self.create_timer(0.1, self.loop)  # 10 Hz
        self.get_logger().info(f'Target height: {self.height} m. Waiting for PX4 to be ready...')

    # ------------------------------------------------------------------ utils
    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def elapsed(self):
        return self.now() - self.state_start

    def set_state(self, state):
        self.get_logger().info(f'[{self.state}] -> [{state}]')
        self.state = state
        self.state_start = self.now()
        self.last_cmd_time = 0.0

    def due(self, period):
        """True at most once per `period` seconds (command retries)."""
        if self.now() - self.last_cmd_time >= period:
            self.last_cmd_time = self.now()
            return True
        return False

    def log_every(self, period, text, warn=False):
        if self.now() - self.last_log_time >= period:
            self.last_log_time = self.now()
            # Two separate calls: rclpy raises an error if one call site logs with different severities
            if warn:
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)

    def armed(self):
        return field(self.status, 'arming_state') == VehicleStatus.ARMING_STATE_ARMED

    def nav_state(self):
        return field(self.status, 'nav_state', -1)

    # -------------------------------------------------------------- callbacks
    def status_cb(self, msg):
        self.status = msg

    def lpos_cb(self, msg):
        self.lpos = msg

    def failsafe_cb(self, msg):
        self.failsafe = msg

    def gps_cb(self, msg):
        self.gps = msg

    # ---------------------------------------------------------- readiness
    def not_ready_reasons(self):
        if self.status is None:
            return ['no vehicle_status (agent not running, or px4_msgs does not match firmware)']

        reasons = []
        if not field(self.status, 'pre_flight_checks_pass', True):
            reasons.append('preflight checks failing (see QGC)')
        if self.lpos is None or not (self.lpos.xy_valid and self.lpos.z_valid):
            reasons.append('local position not valid')

        if REQUIRE_GPS:
            fix = field(self.gps, 'fix_type', 0)
            if fix < 3:
                reasons.append(f'no GPS 3D fix (fix_type={fix})')
            if self.failsafe is None:
                reasons.append('no failsafe_flags yet')
            else:
                if field(self.failsafe, 'global_position_invalid', False):
                    reasons.append('global position invalid')
                if field(self.failsafe, 'home_position_invalid', False):
                    reasons.append('home position not set')
        return reasons

    # ------------------------------------------------------------ main loop
    def loop(self):
        if self.interrupted and self.state not in ('STOP', 'DONE'):
            self.get_logger().warn('Interrupted (Ctrl+C / terminal closed)')
            self.set_state('STOP')

        self.log_flight()

        if self.state == 'WAIT_READY':
            reasons = self.not_ready_reasons()
            if not reasons:
                self.get_logger().info('PX4 ready.')
                self.set_state('ARMING')
            elif self.elapsed() > READY_TIMEOUT:
                self.get_logger().error(f'Not ready after {READY_TIMEOUT}s: {reasons}')
                self.set_state('DONE')
            else:
                self.log_every(3.0, 'Waiting: ' + '; '.join(reasons))

        elif self.state == 'ARMING':
            if self.armed():
                self.z0 = self.lpos.z
                self.set_state('TAKEOFF')
            elif self.elapsed() > ARM_TIMEOUT:
                self.get_logger().error('Arming failed - check QGC messages')
                self.set_state('DONE')
            elif self.due(RESEND_PERIOD):
                self.arm()

        elif self.state == 'TAKEOFF':
            if not self.armed():
                # PX4 auto-disarms if it doesn't take off within COM_DISARM_PRFLT
                self.get_logger().error('Vehicle disarmed before takeoff completed - check QGC messages')
                self.set_state('DONE')
                return
            ns = self.nav_state()
            if ns == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                self.seen_takeoff = True
            if self.seen_takeoff and ns == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                # PX4 switches Takeoff -> Hold when MIS_TAKEOFF_ALT is reached
                self.set_state('CLIMB')
                self.report_height()
            elif self.elapsed() > TAKEOFF_TIMEOUT:
                self.get_logger().error('Takeoff did not complete - landing')
                self.set_state('LAND')
            elif not self.seen_takeoff and self.due(RESEND_PERIOD * 2):
                self.takeoff()

        elif self.state == 'CLIMB':
            height = self.follow_ground()
            if (self.nav_state() == VehicleStatus.NAVIGATION_STATE_POSCTL
                    and abs(height - self.height) < HEIGHT_TOLERANCE):
                self.get_logger().info(f'Reached {height:.2f} m (target {self.height} m) - '
                                       f'hovering {HOVER_TIME:.0f} s')
                self.set_state('HOVER')
            elif self.elapsed() > CLIMB_TIMEOUT:
                self.get_logger().error(f'Height {height:.2f} m, target {self.height} m not reached '
                                        f'after {CLIMB_TIMEOUT:.0f} s - landing')
                self.set_state('LAND')
            else:
                self.request_position_mode()

        elif self.state == 'HOVER':
            self.follow_ground()
            if self.elapsed() >= HOVER_TIME:
                self.set_state('LAND')
            else:
                self.request_position_mode()

        elif self.state == 'LAND':
            if self.armed():
                # Keep the stick (centered) until disarmed: once PX4 has used it, losing it
                # triggers the manual control loss failsafe (NAV_RCL_ACT)
                self.publish_manual_control(0.0)
            # No land-detector topic bridged: PX4 auto-disarms shortly after
            # touchdown (COM_DISARM_LAND), so disarmed == landed.
            if not self.armed():
                self.get_logger().info('Landed and disarmed by PX4.')
                self.set_state('DONE')
            elif self.nav_state() != VehicleStatus.NAVIGATION_STATE_AUTO_LAND and self.due(RESEND_PERIOD * 2):
                self.land()
            else:
                self.log_every(5.0, 'Landing...')

        elif self.state == 'STOP':
            # In Position mode, switch PX4 to Hold before exiting: the node's stick stops with it.
            # In any other mode PX4 just carries on (takeoff, hold, land).
            if not (self.armed() and self.nav_state() == VehicleStatus.NAVIGATION_STATE_POSCTL):
                self.set_state('DONE')
            elif self.elapsed() > STOP_TIMEOUT:
                self.get_logger().error('PX4 did not leave Position mode - manual control loss failsafe will trigger')
                self.set_state('DONE')
            else:
                self.publish_manual_control(0.0)    # centered stick until PX4 is in Hold
                if self.due(RESEND_PERIOD / 2):
                    self.hold_mode()

        elif self.state == 'DONE':
            if self.log_file is not None:
                self.log_file.close()
                self.get_logger().info(f'Flight log saved to {self.log_file.name}')
            self.get_logger().info('>>> Finished. Shutting down node...')
            raise SystemExit

    def log_flight(self):
        """While armed, one csv row per loop (10 Hz): GPS (vehicle_gps_position) and height above the
        ground from the TeraRanger (dist_bottom, empty while it has no valid reading) and from the EKF
        (above the arming point)"""
        if not self.armed() or self.lpos is None or self.state in ('WAIT_READY', 'ARMING'):
            return      # z0 (arming point) is only set once ARMING sees the vehicle armed
        if self.log_file is None:
            folder = os.path.join(LOG_DIR, time.strftime('%Y%m%d_%H%M%S'))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f'{self.get_name()}.csv')
            self.log_file = open(path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(['t', 'state', 'nav_state', 'lat', 'lon', 'alt_msl', 'fix_type', 'satellites',
                                      'eph', 'epv', 'height_teraranger', 'height_ekf', 'target_height'])
            self.log_start = self.now()
            self.get_logger().info(f'Logging GPS and height to {path}')
        sensors = field(self.lpos, 'dist_bottom_sensor_bitfield', 0)
        if field(self.lpos, 'dist_bottom_valid', False) and sensors & VehicleLocalPosition.DIST_BOTTOM_SENSOR_RANGE:
            teraranger = f'{self.lpos.dist_bottom:.3f}'
        else:
            teraranger = ''
        self.log_writer.writerow([
            f'{self.now() - self.log_start:.2f}', self.state, self.nav_state(),
            f"{field(self.gps, 'latitude_deg', NAN):.8f}", f"{field(self.gps, 'longitude_deg', NAN):.8f}",
            f"{field(self.gps, 'altitude_msl_m', NAN):.3f}", field(self.gps, 'fix_type', 0),
            field(self.gps, 'satellites_used', 0), f"{field(self.gps, 'eph', NAN):.2f}",
            f"{field(self.gps, 'epv', NAN):.2f}", teraranger, f'{self.z0 - self.lpos.z:.3f}', self.height])
        self.log_file.flush()   # rows already written are kept even if the node is killed

    def report_height(self):
        if self.lpos is None:
            return
        height = self.z0 - self.lpos.z  # NED: z decreases when climbing
        self.get_logger().info(f'Height above arming point (EKF): {height:.2f} m')
        if field(self.lpos, 'dist_bottom_valid', False):
            self.get_logger().info(f'Distance sensor: {self.lpos.dist_bottom:.2f} m')
        else:
            self.get_logger().warn('Distance sensor: no valid reading')

    def follow_ground(self):
        """Move the throttle stick so the drone climbs/descends to self.height above the ground below it
        (distance sensor), or above the arming point (EKF) while the sensor has no valid reading.
        Inside HEIGHT_TOLERANCE the stick is centered and PX4 holds the altitude.
        Returns the current height."""
        sensors = field(self.lpos, 'dist_bottom_sensor_bitfield', 0)
        if field(self.lpos, 'dist_bottom_valid', False) and sensors & VehicleLocalPosition.DIST_BOTTOM_SENSOR_RANGE:
            height = self.lpos.dist_bottom
            self.log_every(2.0, f'Height above ground: {height:.2f} m (target {self.height} m)')
        else:
            height = self.z0 - self.lpos.z  # NED: z decreases when climbing
            self.log_every(2.0, 'Distance sensor: no valid reading - height from EKF: '
                                f'{height:.2f} m above the arming point (target {self.height} m)',
                           warn=True)
        error = self.height - height
        throttle = 0.0
        # Stick only in Position mode: in Hold, a moved stick makes PX4 switch to Position itself (COM_RC_OVERRIDE)
        if self.nav_state() == VehicleStatus.NAVIGATION_STATE_POSCTL and abs(error) >= HEIGHT_TOLERANCE:
            throttle = math.copysign(min(STICK_MAX, max(STICK_MIN, STICK_GAIN * abs(error))), error)
        self.publish_manual_control(throttle)
        return height

    def request_position_mode(self):
        # PX4 only accepts Position mode once it has a valid stick input (streaming for a while)
        if (self.nav_state() != VehicleStatus.NAVIGATION_STATE_POSCTL
                and self.elapsed() > 1.0 and self.due(RESEND_PERIOD)):
            self.set_position_mode()

    def interrupt(self, signum, frame):
        """Signal handler (Ctrl+C / terminal closed): only set a flag, loop() switches to STOP"""
        self.interrupted = True

    # ------------------------------------------------------------- commands
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f'>>> Sent ARM to Vehicle {TARGET_SYSTEM}')

    def takeoff(self):
        # param4 yaw, param5 lat, param6 lon, param7 altitude AMSL.
        # All NaN -> take off at current position/heading to MIS_TAKEOFF_ALT above home
        # (same as MAVSDK action.takeoff()).
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            param4=NAN, param5=NAN, param6=NAN, param7=NAN)
        self.get_logger().info('>>> Sent TAKEOFF (altitude = MIS_TAKEOFF_ALT)')

    def land(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
            param4=NAN, param5=NAN, param6=NAN, param7=NAN)
        self.get_logger().info('>>> Sent LAND')

    def set_position_mode(self):
        # param1 = 1 (custom mode), param2 = 3 (PX4 POSCTL = Position mode)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=3.0)
        self.get_logger().info('>>> Sent POSITION mode')

    def hold_mode(self):
        # param1 = 1 (custom mode), param2 = 4 (AUTO), param3 = 3 (LOITER = Hold)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0, param3=3.0)
        self.get_logger().info('>>> Sent HOLD mode')

    def publish_manual_control(self, throttle):
        msg = ManualControlSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp    # PX4 drops the input COM_RC_LOSS_T after this
        msg.valid = True
        # PX4 has no DDS source: use a MAVLink instance QGC (virtual joystick) does not use
        msg.data_source = ManualControlSetpoint.SOURCE_MAVLINK_5
        msg.roll = 0.0          # centered: PX4 holds the GPS position
        msg.pitch = 0.0
        msg.yaw = 0.0           # centered: PX4 holds the heading
        msg.throttle = float(throttle)  # [-1, 1], 0 = hold altitude
        for pub in self.manual_pubs:
            pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0,
                                param4=0.0, param5=0.0, param6=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.target_system = TARGET_SYSTEM
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)


def main(args=None):
    height = float(sys.argv[1]) if len(sys.argv) > 1 else HOLD_HEIGHT
    if height <= 0:
        sys.exit(f'Height must be > 0 m (got {height})')
    # Ctrl+C (SIGINT), closing the terminal (SIGHUP) or kill (SIGTERM) must not stop the node right away:
    # ROS2 keeps running and loop() first switches PX4 from Position mode to Hold (state STOP)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TakeoffLandMissionPosition(height)
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, node.interrupt)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
