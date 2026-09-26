#!/usr/bin/env python3
"""
takeoff_land_mission.py  (ROS2 + PX4 uXRCE-DDS)

Takeoff -> hover -> land, driven by vehicle state (same flow as the tested
MAVSDK script takeoff_and_land_classic.py).

Uses ONLY topics bridged by this drone's current dds_topics.yaml:
  in : /fmu/in/vehicle_command
  out: /fmu/out/vehicle_status, /fmu/out/vehicle_local_position,
       /fmu/out/failsafe_flags, /fmu/out/vehicle_gps_position

Flow:
  1. Wait: vehicle_status received, preflight checks pass, local position valid,
     and (if REQUIRE_GPS) GPS 3D fix + global position + home position valid
  2. Arm, retry until vehicle_status says ARMED
  3. NAV_TAKEOFF with NaN params -> PX4 climbs to MIS_TAKEOFF_ALT above home
  4. When PX4 switches Takeoff -> Hold: hover HOVER_TIME s, print height/distance sensor
  5. NAV_LAND, wait for PX4 auto-disarm after touchdown (no manual disarm)

Set MIS_TAKEOFF_ALT (e.g. 1.5) in QGroundControl before flying.
px4_msgs must match the PX4 firmware version, otherwise subscriptions fail
with "Fast CDR exception deserializing ...".
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
    FailsafeFlags,
    SensorGps,
)

NAN = float('nan')

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TARGET_SYSTEM = 10        # must match MAV_SYS_ID on the flight controller
REQUIRE_GPS = True        # outdoor: wait for GPS/home like ensure_gps_estimate(); False indoors
HOVER_TIME = 10.0         # s to hover after takeoff completes

READY_TIMEOUT = 120.0     # s to wait for PX4 to be ready
ARM_TIMEOUT = 10.0        # s to wait for ARMED
TAKEOFF_TIMEOUT = 30.0    # s to wait for takeoff to finish (then land)
RESEND_PERIOD = 1.0       # s between command retries

TOPIC_CMD = '/fmu/in/vehicle_command'
TOPIC_STATUS = '/fmu/out/vehicle_status'
TOPIC_LPOS = '/fmu/out/vehicle_local_position'
TOPIC_FAILSAFE = '/fmu/out/failsafe_flags'
TOPIC_GPS = '/fmu/out/vehicle_gps_position'


def field(msg, name, default=None):
    """Read a message field safely (field names can differ between PX4 versions)."""
    return getattr(msg, name, default) if msg is not None else default


class TakeoffLandMission(Node):
    def __init__(self):
        super().__init__('takeoff_land_mission')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_pub = self.create_publisher(VehicleCommand, TOPIC_CMD, qos)

        self.create_subscription(VehicleStatus, TOPIC_STATUS, self.status_cb, qos)
        self.create_subscription(VehicleLocalPosition, TOPIC_LPOS, self.lpos_cb, qos)
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

        self.timer = self.create_timer(0.1, self.loop)  # 10 Hz
        self.get_logger().info('Waiting for PX4 to be ready...')

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
            (self.get_logger().warn if warn else self.get_logger().info)(text)

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
                self.set_state('HOVER')
                self.report_height()
            elif self.elapsed() > TAKEOFF_TIMEOUT:
                self.get_logger().error('Takeoff did not complete - landing')
                self.set_state('LAND')
            elif not self.seen_takeoff and self.due(RESEND_PERIOD * 2):
                self.takeoff()

        elif self.state == 'HOVER':
            if self.elapsed() >= HOVER_TIME:
                self.set_state('LAND')

        elif self.state == 'LAND':
            # No land-detector topic bridged: PX4 auto-disarms shortly after
            # touchdown (COM_DISARM_LAND), so disarmed == landed.
            if not self.armed():
                self.get_logger().info('Landed and disarmed by PX4.')
                self.set_state('DONE')
            elif self.nav_state() != VehicleStatus.NAVIGATION_STATE_AUTO_LAND and self.due(RESEND_PERIOD * 2):
                self.land()
            else:
                self.log_every(5.0, 'Landing...')

        elif self.state == 'DONE':
            self.get_logger().info('>>> Finished. Shutting down node...')
            raise SystemExit

    def report_height(self):
        if self.lpos is None:
            return
        height = self.z0 - self.lpos.z  # NED: z decreases when climbing
        self.get_logger().info(f'Height above arming point (EKF): {height:.2f} m')
        if field(self.lpos, 'dist_bottom_valid', False):
            self.get_logger().info(f'Distance sensor: {self.lpos.dist_bottom:.2f} m')
        else:
            self.get_logger().warn('Distance sensor: no valid reading')

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
    rclpy.init(args=args)
    node = TakeoffLandMission()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()