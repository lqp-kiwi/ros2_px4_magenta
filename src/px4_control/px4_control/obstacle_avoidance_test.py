#!/usr/bin/env python3
"""
    Simple obstacle avoidance test for PX4 using ROS2 with GPS, flown in OFFBOARD.
    1. Takes off, then climbs/descends in OFFBOARD to the height given as argument above the ground
       right below the drone, measured by the downward distance sensor (vehicle_local_position.dist_bottom),
       or above the arming point (EKF) while the sensor has no valid reading.
    2. Holds there. From now on, anything the depth camera sees closer than NEAR_DIST makes the drone
       fly DODGE_DIST straight backward, opposite to where the camera looks.
       It holds the new position (it does not go forward again) and flies back again if the obstacle is still near.
       Height above the ground and heading stay the same. The camera only looks forward: an obstacle
       coming from the side or from behind is not seen, also not the one the drone backs into.
      - Ctrl+C / closing the terminal switches PX4 to Hold before the node exits
      - switching PX4 to another mode (RC / QGC) ends the node: the drone is the pilot's
    Land with RC or QGC.
    Author: Phuong Le (lephuo10@rowan.edu)
    Last Updated: 2026-11-09
    Position, height and obstacle distance are logged to flight_logs/<date>_<time>/<node>.csv while flying.
    Needs /camera/depth/image_raw and /camera/depth/camera_info, the node does not take off without them:
        real drone: ros2 run depth_camera camera --no-cloud
        simulation: ros2 run ros_gz_bridge parameter_bridge "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image" \
                        "/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" --ros-args \
                        -r /depth_camera:=/camera/depth/image_raw -r /camera_info:=/camera/depth/camera_info
    Usage:
	    ros2 run px4_control obstacle_avoidance_test [height_m]     (default HOLD_HEIGHT)
"""

import csv
import math
import os
import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, CameraInfo
from px4_msgs.msg import (
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
    FailsafeFlags,
    SensorGps,
    OffboardControlMode,
    TrajectorySetpoint,
)

NAN = float('nan')

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
REQUIRE_GPS = True        # outdoor: wait for GPS/home like ensure_gps_estimate(); False indoors
HOLD_HEIGHT = 3.0         # m, target height when no argument is given: above the ground below the
                          # drone (distance sensor), above the arming point (EKF) while it has no valid reading
HOLD_CLIMB = 0.3          # m/s, max speed of the height setpoint (no jumps)
HEIGHT_TOLERANCE = 0.15   # m, target height counts as reached within this error

# Obstacle detection
NEAR_DIST = 2.5           # m, anything closer than this (depth along the camera axis) is an obstacle
NEAR_FRACTION = 0.003     # share of the image that must be closer than NEAR_DIST, less is depth noise
NEAR_FRAMES = 2           # depth images in a row with an obstacle before dodging
DEPTH_STRIDE = 4          # use every 4th pixel
DEPTH_MIN = 0.5           # m, nearer pixels are ignored (no data, parts of the drone)
GROUND_IGNORE = 0.5       # m, ignore points lower than this above the ground
DEPTH_TIMEOUT = 0.5       # s, no depth image for this long -> warning, does not take off

# Dodge: straight backward
DODGE_DIST = 1.5          # m, moved back per dodge
DODGE_SPEED = 0.3         # m/s, speed of a dodge
DODGE_REACHED = 0.3       # m, dodge is finished this close to its target
DODGE_TIMEOUT = 8.0       # s, dodge is finished after this time at the latest (DODGE_DIST / DODGE_SPEED = 5 s)
MAX_OFFSET = 5.0          # m, never dodge farther than this from the first hold point (geofence)

LOOP_PERIOD = 0.05        # s, main loop and offboard setpoints (20 Hz)
READY_TIMEOUT = 120.0     # s to wait for PX4 to be ready
ARM_TIMEOUT = 10.0        # s to wait for ARMED
TAKEOFF_TIMEOUT = 30.0    # s to wait for takeoff to finish (then land)
CLIMB_TIMEOUT = 60.0      # s to reach the target height after takeoff (then land)
STOP_TIMEOUT = 3.0        # s to wait for PX4 to leave OFFBOARD after Ctrl+C / terminal closed
RESEND_PERIOD = 1.0       # s between command retries
LOG_DIR = 'flight_logs'   # csv per flight, relative to the folder the node is started from

TOPIC_CMD = '/fmu/in/vehicle_command'
TOPIC_OFFBOARD = '/fmu/in/offboard_control_mode'
TOPIC_SETPOINT = '/fmu/in/trajectory_setpoint'
TOPIC_STATUS = '/fmu/out/vehicle_status'
TOPIC_LPOS = '/fmu/out/vehicle_local_position'
TOPIC_FAILSAFE = '/fmu/out/failsafe_flags'
TOPIC_GPS = '/fmu/out/vehicle_gps_position'
TOPIC_DEPTH = '/camera/depth/image_raw'
TOPIC_CAMERA_INFO = '/camera/depth/camera_info'


def field(msg, name, default=None):
    """Read a message field safely (field names can differ between PX4 versions)."""
    return getattr(msg, name, default) if msg is not None else default


def versioned(topic, msg_type):
    """PX4 v1.16+ adds _v<MESSAGE_VERSION> to the topic name of versioned messages
    (e.g. vehicle_status_v4). Older PX4 uses the plain name."""
    version = getattr(msg_type, 'MESSAGE_VERSION', 0)
    return f'{topic}_v{version}' if version else topic


def depth_to_meters(msg):
    """Depth image -> 2D numpy array in meters (16UC1/mono16 = mm, 32FC1 = m)"""
    if msg.encoding in ('16UC1', 'mono16'):
        dtype = '>u2' if msg.is_bigendian else '<u2'
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return img.astype(np.float32) / 1000.0
    if msg.encoding == '32FC1':
        dtype = '>f4' if msg.is_bigendian else '<f4'
        return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 4)[:, :msg.width]
    raise ValueError(f'Unsupported depth encoding: {msg.encoding}')


def find_obstacle(depth, row_slope, height):
    """Looks at the depth image without the ground. row_slope: meters below the camera axis per meter
    of depth, for each row. height: height of the camera above the ground (m).
    Returns (nearest, found): the nearest point seen (m, inf = nothing) and whether enough of the image
    is closer than NEAR_DIST."""
    with np.errstate(invalid='ignore'):
        seen = (depth > DEPTH_MIN) & (height - depth * row_slope > GROUND_IGNORE)
        near = seen & (depth < NEAR_DIST)
    nearest = float(depth[seen].min()) if seen.any() else math.inf
    return nearest, near.mean() >= NEAR_FRACTION


class ObstacleAvoidanceTest(Node):
    def __init__(self, height):
        super().__init__('obstacle_avoidance_test')
        self.height = height      # m, target height above the ground

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_pub = self.create_publisher(VehicleCommand, TOPIC_CMD, qos)
        self.offboard_pub = self.create_publisher(OffboardControlMode, TOPIC_OFFBOARD, qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, TOPIC_SETPOINT, qos)

        # Listen to both the plain and the versioned name: works with old and new PX4
        for topic in {TOPIC_STATUS, versioned(TOPIC_STATUS, VehicleStatus)}:
            self.create_subscription(VehicleStatus, topic, self.status_cb, qos)
        for topic in {TOPIC_LPOS, versioned(TOPIC_LPOS, VehicleLocalPosition)}:
            self.create_subscription(VehicleLocalPosition, topic, self.lpos_cb, qos)
        self.create_subscription(FailsafeFlags, TOPIC_FAILSAFE, self.failsafe_cb, qos)
        self.create_subscription(SensorGps, TOPIC_GPS, self.gps_cb, qos)
        self.create_subscription(Image, TOPIC_DEPTH, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, TOPIC_CAMERA_INFO, self.camera_info_cb, qos_profile_sensor_data)

        self.status = None
        self.lpos = None
        self.failsafe = None
        self.gps = None
        self.camera_info = None
        self.depth_time = 0.0     # time of the newest depth image
        self.nearest = math.inf   # m, nearest point in the newest depth image, without the ground
        self.near_frames = 0      # depth images in a row with an obstacle

        self.state = 'WAIT_READY'
        self.state_start = self.now()
        self.last_cmd_time = 0.0
        self.last_log_time = 0.0
        self.seen_takeoff = False
        self.z0 = 0.0
        self.hold = None          # [x, y, z, yaw] offboard setpoint while hovering (NED)
        self.dodge_target = None  # (x, y) the current dodge flies to
        self.home_xy = None       # first hold point (geofence center)
        self.interrupted = False  # set by Ctrl+C / terminal closed, handled in loop()
        self.log_file = None      # csv, opened once flying (log_flight)
        self.log_writer = None
        self.log_start = 0.0

        self.timer = self.create_timer(LOOP_PERIOD, self.loop)
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

    def offboard(self):
        return self.armed() and self.nav_state() == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def height_above_ground(self):
        """(height, True) from the distance sensor,
        (height above the arming point from the EKF, False) while it has no valid reading"""
        sensors = field(self.lpos, 'dist_bottom_sensor_bitfield', 0)
        if field(self.lpos, 'dist_bottom_valid', False) and sensors & VehicleLocalPosition.DIST_BOTTOM_SENSOR_RANGE:
            return self.lpos.dist_bottom, True
        return self.z0 - self.lpos.z, False  # NED: z decreases when climbing

    # -------------------------------------------------------------- callbacks
    def status_cb(self, msg):
        self.status = msg

    def lpos_cb(self, msg):
        self.lpos = msg

    def failsafe_cb(self, msg):
        self.failsafe = msg

    def gps_cb(self, msg):
        self.gps = msg

    def camera_info_cb(self, msg):
        self.camera_info = msg

    def depth_cb(self, msg):
        self.depth_time = self.now()
        if self.camera_info is None or self.lpos is None:
            return
        fy, cy = self.camera_info.k[4], self.camera_info.k[5]
        depth = depth_to_meters(msg)[::DEPTH_STRIDE, ::DEPTH_STRIDE]
        row_slope = (np.arange(depth.shape[0])[:, None] * DEPTH_STRIDE - cy) / fy
        self.nearest, found = find_obstacle(depth, row_slope, self.height_above_ground()[0])
        self.near_frames = self.near_frames + 1 if found else 0
        if self.state == 'HOLD' and self.near_frames >= NEAR_FRAMES:
            self.start_dodge()

    # ---------------------------------------------------------- readiness
    def not_ready_reasons(self):
        if self.status is None:
            return ['no vehicle_status (agent not running, or px4_msgs does not match firmware)']

        reasons = []
        if not field(self.status, 'pre_flight_checks_pass', True):
            reasons.append('preflight checks failing (see QGC)')
        if self.lpos is None or not (self.lpos.xy_valid and self.lpos.z_valid):
            reasons.append('local position not valid')
        if self.camera_info is None or self.now() - self.depth_time > DEPTH_TIMEOUT:
            reasons.append('no depth image / camera_info (camera node or bridge not running?)')

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
                self.hold = [self.lpos.x, self.lpos.y, self.lpos.z, self.lpos.heading]
            elif self.elapsed() > TAKEOFF_TIMEOUT:
                self.get_logger().error('Takeoff did not complete - landing')
                self.set_state('LAND')
            elif not self.seen_takeoff and self.due(RESEND_PERIOD * 2):
                self.takeoff()

        elif self.state == 'CLIMB':
            height = self.follow_ground()
            if self.offboard() and abs(height - self.height) < HEIGHT_TOLERANCE:
                self.get_logger().info(f'Reached {height:.2f} m (target {self.height} m) - obstacle avoidance ON, '
                                       'holding until Ctrl+C or another mode from RC / QGC')
                self.home_xy = (self.lpos.x, self.lpos.y)
                self.set_state('HOLD')
            elif self.elapsed() > CLIMB_TIMEOUT:
                self.get_logger().error(f'Height {height:.2f} m, target {self.height} m not reached '
                                        f'after {CLIMB_TIMEOUT:.0f} s - landing')
                self.set_state('LAND')
            else:
                self.request_offboard()

        elif self.state in ('HOLD', 'DODGE'):
            if not self.offboard():
                # Pilot / QGC switched mode, a PX4 failsafe or disarmed: leave the drone to PX4, do not take it back
                self.get_logger().warn(f'PX4 left OFFBOARD or disarmed (nav_state {self.nav_state()}) - stopping')
                self.set_state('DONE')
            elif self.state == 'DODGE':
                self.fly_dodge()
            else:
                self.follow_ground()
                if self.now() - self.depth_time > DEPTH_TIMEOUT:
                    self.log_every(2.0, 'No depth image - cannot see obstacles', warn=True)

        elif self.state == 'LAND':
            if self.offboard():
                # Keep streaming until PX4 is in Land, otherwise the offboard-loss failsafe triggers
                # (after landing PX4 shows OFFBOARD again, but then it is disarmed)
                self.follow_ground()
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
            # In OFFBOARD, switch PX4 to Hold before exiting: when the setpoints stop, PX4 triggers the
            # offboard-loss failsafe (COM_OBL_RC_ACT, Return mode when there is no RC).
            # In any other mode PX4 just carries on (takeoff, hold, land).
            if not self.offboard():
                self.set_state('DONE')
            elif self.elapsed() > STOP_TIMEOUT:
                self.get_logger().error('PX4 did not leave OFFBOARD - offboard-loss failsafe will trigger')
                self.set_state('DONE')
            else:
                self.follow_ground()    # keep streaming until PX4 is in Hold
                if self.due(RESEND_PERIOD / 2):
                    self.hold_mode()

        elif self.state == 'DONE':
            if self.log_file is not None:
                self.log_file.close()
                self.get_logger().info(f'Flight log saved to {self.log_file.name}')
            self.get_logger().info('>>> Finished. Shutting down node...')
            raise SystemExit

    def log_flight(self):
        """While armed, one csv row per loop: local position (NED), height above the ground from the distance
        sensor (empty while it has no valid reading) and from the EKF (above the arming point), setpoint,
        nearest point seen by the depth camera without the ground (empty = nothing)"""
        if not self.armed() or self.lpos is None or self.state in ('WAIT_READY', 'ARMING'):
            return      # z0 (arming point) is only set once ARMING sees the vehicle armed
        if self.log_file is None:
            folder = os.path.join(LOG_DIR, time.strftime('%Y%m%d_%H%M%S'))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f'{self.get_name()}.csv')
            self.log_file = open(path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(['t', 'state', 'nav_state', 'x', 'y', 'z', 'height_sensor', 'height_ekf',
                                      'target_height', 'sp_x', 'sp_y', 'sp_z', 'nearest'])
            self.log_start = self.now()
            self.get_logger().info(f'Logging to {path}')
        height, from_sensor = self.height_above_ground()
        sp = [f'{v:.3f}' for v in self.hold[:3]] if self.hold is not None else ['', '', '']
        self.log_writer.writerow([
            f'{self.now() - self.log_start:.2f}', self.state, self.nav_state(),
            f'{self.lpos.x:.3f}', f'{self.lpos.y:.3f}', f'{self.lpos.z:.3f}',
            f'{height:.3f}' if from_sensor else '', f'{self.z0 - self.lpos.z:.3f}', self.height,
            *sp, f'{self.nearest:.3f}' if math.isfinite(self.nearest) else ''])
        self.log_file.flush()   # rows already written are kept even if the node is killed

    def follow_ground(self):
        """Hold x, y, heading and set z so the drone stays self.height above the ground below it
        (distance sensor), or above the arming point (EKF) while the sensor has no valid reading.
        The z setpoint moves at most HOLD_CLIMB, so a new target does not make the drone jump.
        Returns the current height used for the setpoint."""
        height, from_sensor = self.height_above_ground()
        if from_sensor:
            target_z = self.lpos.z + height - self.height  # NED: z grows downwards
            self.log_every(2.0, f'Height above ground: {height:.2f} m (target {self.height} m), '
                                f'nearest obstacle: {self.nearest:.2f} m')
        else:
            target_z = self.z0 - self.height
            self.log_every(2.0, 'Distance sensor: no valid reading - height from EKF: '
                                f'{height:.2f} m above the arming point (target {self.height} m)',
                           warn=True)
        step = HOLD_CLIMB * LOOP_PERIOD
        self.hold[2] += max(-step, min(step, target_z - self.hold[2]))
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(*self.hold, (NAN, NAN, NAN))
        return height

    def start_dodge(self):
        """Move the hold point DODGE_DIST straight back from where the drone is now, opposite to the camera"""
        yaw = self.hold[3]
        north = self.lpos.x - DODGE_DIST * math.cos(yaw)
        east = self.lpos.y - DODGE_DIST * math.sin(yaw)
        # Geofence: stay within MAX_OFFSET of the first hold point
        d_north, d_east = north - self.home_xy[0], east - self.home_xy[1]
        offset = math.hypot(d_north, d_east)
        if offset > MAX_OFFSET:
            north = self.home_xy[0] + d_north * MAX_OFFSET / offset
            east = self.home_xy[1] + d_east * MAX_OFFSET / offset
        if math.hypot(north - self.lpos.x, east - self.lpos.y) < DODGE_REACHED:
            self.log_every(2.0, f'Obstacle at {self.nearest:.2f} m but MAX_OFFSET reached - cannot dodge', warn=True)
            return
        self.hold[0], self.hold[1] = self.lpos.x, self.lpos.y   # fly_dodge() moves it to the target
        self.dodge_target = (north, east)
        self.get_logger().info(f'>>> Obstacle at {self.nearest:.2f} m -> flying back {DODGE_DIST} m')
        self.set_state('DODGE')

    def fly_dodge(self):
        """Move the setpoint to the target at DODGE_SPEED: with the setpoint far ahead PX4 would fly much faster"""
        d_north, d_east = self.dodge_target[0] - self.hold[0], self.dodge_target[1] - self.hold[1]
        left = math.hypot(d_north, d_east)
        step = DODGE_SPEED * LOOP_PERIOD
        if left > step:
            self.hold[0] += d_north * step / left
            self.hold[1] += d_east * step / left
        else:
            self.hold[0], self.hold[1] = self.dodge_target
        reached = left <= step and math.hypot(self.hold[0] - self.lpos.x, self.hold[1] - self.lpos.y) < DODGE_REACHED
        if reached or self.elapsed() > DODGE_TIMEOUT:
            self.get_logger().info(f'Dodge finished, nearest obstacle: {self.nearest:.2f} m')
            self.set_state('HOLD')
        self.follow_ground()

    def request_offboard(self):
        # PX4 only accepts OFFBOARD after setpoints have been streaming for a while
        if (self.nav_state() != VehicleStatus.NAVIGATION_STATE_OFFBOARD
                and self.elapsed() > 1.0 and self.due(RESEND_PERIOD)):
            self.set_offboard_mode()

    def interrupt(self, signum, frame):
        """Signal handler (Ctrl+C / terminal closed): only set a flag, loop() switches to STOP"""
        self.interrupted = True

    # ------------------------------------------------------------- commands
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f'>>> Sent ARM to Vehicle {self.status.system_id}')

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

    def set_offboard_mode(self):
        # param1 = 1 (custom mode), param2 = 6 (PX4 OFFBOARD mode)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('>>> Sent OFFBOARD mode')

    def hold_mode(self):
        # param1 = 1 (custom mode), param2 = 4 (AUTO), param3 = 3 (LOITER = Hold)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0, param3=3.0)
        self.get_logger().info('>>> Sent HOLD mode')

    # Messages to PX4 have timestamp = 0: PX4 stamps them itself when they arrive. With the ROS clock PX4 has to
    # convert it with its estimate of the clock difference, which can be off by seconds when the simulation runs
    # slower than real time: PX4 then treats the offboard stream as lost (OFFBOARD refused, or failsafe).
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True     # control by position setpoint (velocity is only feedforward)
        msg.timestamp = 0
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw, vel):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float(v) for v in vel]
        msg.acceleration = [NAN, NAN, NAN]
        msg.yaw = float(yaw)
        msg.timestamp = 0
        self.setpoint_pub.publish(msg)

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
        msg.target_system = self.status.system_id   # the flight controller's own MAV_SYS_ID (sim 1, drone 10)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = 0
        self.cmd_pub.publish(msg)


def main(args=None):
    height = float(sys.argv[1]) if len(sys.argv) > 1 else HOLD_HEIGHT
    if height <= 0:
        sys.exit(f'Height must be > 0 m (got {height})')
    # Ctrl+C (SIGINT), closing the terminal (SIGHUP) or kill (SIGTERM) must not stop the node right away:
    # ROS2 keeps running and loop() first switches PX4 from OFFBOARD to Hold (state STOP)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ObstacleAvoidanceTest(height)
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
