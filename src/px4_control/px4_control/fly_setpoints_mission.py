#!/usr/bin/env python3

import sys
import json
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus, FailsafeFlags, SensorGps)

"""
Script Name: fly_states.py
Description: This script reads a .json file made by record_drone_state.py and flies
             the drone (PX4 offboard mode over ROS2) through the saved states:
             take off to TAKEOFF_HEIGHT -> state 2 ... state N -> state 1 (home) -> land.
             All states are flown at the same height above the takeoff ground.

Last Updated: 2026-10-03

Usage:
	ros2 run px4_control fly_setpoints_mission [filename]
"""

TARGET_SYSTEM = 10      # must match MAV_SYS_ID on the flight controller
TAKEOFF_HEIGHT = 5.0    # meters above the ground at the takeoff position
ACCEPT_RADIUS = 1.0     # meters, a state counts as reached inside this distance
HOLD_TIME = 2.0         # seconds to hover at each state (also gives time to turn)
GOTO_TIMEOUT = 60.0     # seconds to reach a state, otherwise land
MAX_DISTANCE = 200.0    # meters, refuse to fly if a state is farther than this from the drone
EARTH_RADIUS = 6371000.0
NAN = float('nan')

class PilotTookOver(Exception):
	pass

def gps_to_local(lat, lon, ref_lat, ref_lon):
	# Convert GPS (degrees) to local North/East meters relative to the EKF origin
	x = math.radians(lat - ref_lat) * EARTH_RADIUS
	y = math.radians(lon - ref_lon) * EARTH_RADIUS * math.cos(math.radians(ref_lat))
	return x, y

class FlyStates(Node):
	def __init__(self):
		super().__init__("fly_states")
		qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
		                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
		                 history=HistoryPolicy.KEEP_LAST, depth=1)
		self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
		self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
		self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
		self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.save_position, qos)
		self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.save_status, qos)
		self.create_subscription(FailsafeFlags, "/fmu/out/failsafe_flags", self.save_failsafe, qos)
		self.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", self.save_gps, qos)
		self.position = None
		self.status = None
		self.failsafe = None
		self.gps = None
		self.target = None  # (x, y, z, yaw) in local NED, sent to PX4 10 times per second
		self.create_timer(0.1, self.publish_setpoint)

	def save_position(self, msg):
		self.position = msg

	def save_status(self, msg):
		self.status = msg

	def save_failsafe(self, msg):
		self.failsafe = msg

	def save_gps(self, msg):
		self.gps = msg

	def now_us(self):
		return int(self.get_clock().now().nanoseconds / 1000)

	def publish_setpoint(self):
		# PX4 leaves offboard mode if these messages stop, so they are sent continuously
		if self.target is None:
			return
		mode = OffboardControlMode()
		mode.position = True
		mode.timestamp = self.now_us()
		self.offboard_pub.publish(mode)

		x, y, z, yaw = self.target
		setpoint = TrajectorySetpoint()
		setpoint.position = [float(x), float(y), float(z)]
		setpoint.velocity = [NAN] * 3
		setpoint.acceleration = [NAN] * 3
		setpoint.yaw = float(yaw)
		setpoint.yawspeed = NAN
		setpoint.timestamp = self.now_us()
		self.setpoint_pub.publish(setpoint)

	def send_command(self, command, param1=0.0, param2=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0):
		msg = VehicleCommand()
		msg.command = command
		msg.param1 = float(param1)
		msg.param2 = float(param2)
		msg.param4 = float(param4)
		msg.param5 = float(param5)
		msg.param6 = float(param6)
		msg.param7 = float(param7)
		msg.target_system = TARGET_SYSTEM
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.now_us()
		self.command_pub.publish(msg)

	def is_armed(self):
		return self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

	def is_offboard(self):
		return self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

	def not_ready_reasons(self):
		# Same checks as the tested takeoff_land_mission.py
		if self.status is None:
			return ["no vehicle_status"]
		reasons = []
		if not getattr(self.status, "pre_flight_checks_pass", True):
			reasons.append("preflight checks failing (see QGC)")
		p = self.position
		if p is None or not (p.xy_valid and p.z_valid and p.xy_global):
			reasons.append("local/global position not valid")
		fix = getattr(self.gps, "fix_type", 0)
		if fix < 3:
			reasons.append(f"no GPS 3D fix (fix_type={fix})")
		if self.failsafe is None:
			reasons.append("no failsafe_flags yet")
		else:
			if getattr(self.failsafe, "global_position_invalid", False):
				reasons.append("global position invalid")
			if getattr(self.failsafe, "home_position_invalid", False):
				reasons.append("home position not set")
		return reasons

def load_states():
	if len(sys.argv) > 1:
		filename = sys.argv[1]
	else:
		filename = input("Enter the name of the .json file with the saved states:")
	if filename.find(".") == -1:
		filename = filename + ".json"
	with open(filename) as f:
		states = json.load(f)
	print(f"Loaded {len(states)} state(s) from {filename}")
	return states

def goto(node, target, name):
	# Send the drone to target and wait until it is within ACCEPT_RADIUS
	node.target = target
	print(f"Flying to {name}...")
	start_time = time.time()
	while True:
		if not node.is_armed() or not node.is_offboard():
			raise PilotTookOver()
		p = node.position
		if math.dist((p.x, p.y, p.z), target[:3]) < ACCEPT_RADIUS:
			break
		if time.time() - start_time > GOTO_TIMEOUT:
			raise TimeoutError(f"could not reach {name} in {GOTO_TIMEOUT} s")
		time.sleep(0.2)
	print(f"-- Reached {name}")
	time.sleep(HOLD_TIME)

def land(node):
	# Resend LAND until PX4 is in land mode (commands can be lost), then wait for auto-disarm
	node.target = None
	print("Landing...")
	while node.is_armed():
		if node.status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
			node.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, param4=NAN, param5=NAN, param6=NAN, param7=NAN)
		time.sleep(1.0)
	print("-- Landed and disarmed")

def main():
	states = load_states()
	if len(states) < 2:
		print("Need at least 2 states (state 1 = home).")
		return
        
	rclpy.init()
	node = FlyStates()
	threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

	print("Waiting for PX4 to be ready...")
	while True:
		reasons = node.not_ready_reasons()
		if not reasons:
			break
		print("Waiting: " + "; ".join(reasons))
		time.sleep(2.0)
	print("-- PX4 ready")

	# Convert every state to local coordinates, all at TAKEOFF_HEIGHT above the current ground
	start = node.position
	ground_z = start.z
	fly_z = ground_z - TAKEOFF_HEIGHT  # NED: up is negative z
	waypoints = []
	for i, state in enumerate(states):
		x, y = gps_to_local(state["global_position"]["lat"], state["global_position"]["lon"],
		                    start.ref_lat, start.ref_lon)
		yaw = state["local_position"].get("heading", NAN)
		distance = math.dist((x, y), (start.x, start.y))
		print(f"State {i + 1}: {distance:.1f} m from the drone, heading {math.degrees(yaw):.0f} deg")
		if distance > MAX_DISTANCE:
			print(f"State {i + 1} is farther than {MAX_DISTANCE} m. Check the file. Aborting.")
			return
		waypoints.append((x, y, fly_z, yaw))

	# Route: state 2 ... state N, then back to state 1 (home)
	route = [(f"state {i + 1}", wp) for i, wp in enumerate(waypoints)][1:] + [("home (state 1)", waypoints[0])]

	input(f"Press Enter to arm and take off to {TAKEOFF_HEIGHT} m (Ctrl+C to cancel)")
	try:
		# Hold the current position, then switch to offboard and arm
		node.target = (start.x, start.y, ground_z, start.heading)
		time.sleep(1.0)
		for _ in range(10):
			if node.is_armed() and node.is_offboard():
				break
			node.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)  # 6 = offboard
			node.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)  # 1 = arm
			time.sleep(1.0)
		else:
			print("Could not arm / switch to offboard. Check QGroundControl for the reason.")
			node.target = None
			return
		print("-- Armed in offboard mode")

		goto(node, (start.x, start.y, fly_z, start.heading), f"takeoff height {TAKEOFF_HEIGHT} m")
		for name, wp in route:
			goto(node, wp, name)
		land(node)
	except PilotTookOver:
		# Mode changed from RC/QGC (or disarmed): stop sending anything, the pilot is in control
		node.target = None
		print("Left offboard mode or disarmed - pilot has control, script stopped.")
	except TimeoutError as e:
		print(f"{e}, landing now.")
		land(node)
	except KeyboardInterrupt:
		print("\nCtrl+C pressed, landing now.")
		land(node)

	node.destroy_node()
	rclpy.shutdown()

if __name__ == "__main__":
	main()