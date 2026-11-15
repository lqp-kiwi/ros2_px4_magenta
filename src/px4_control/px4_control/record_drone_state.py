#!/usr/bin/env python3

import sys
import os
import json
import time
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from px4_msgs.msg import VehicleGlobalPosition, VehicleLocalPosition, VehicleAttitude, VehicleStatus

from px4_control.gps_point import GPS_POINT_FILE

"""
Script Name: record_drone_state.py
Description: This script creates a .json file of the drone's current state,
             saving the raw px4_msgs messages returned by the uXRCE-DDS agent.
             Without a filename it saves one GPS point to GPS_POINT_FILE (gps_point.py), the point the
             obstacle_*.py scripts fly to. Saving again replaces it.
             With a filename it saves every state taken in this run to that file (fly_setpoints_mission.py).

Last Modified: 2026-11-09

Usage:
	ros2 run px4_control record_drone_state [filename]
"""

# Topics to record: name in the file -> (topic, message type)
TOPICS = {
	"global_position": ("/fmu/out/vehicle_global_position", VehicleGlobalPosition),
	"local_position": ("/fmu/out/vehicle_local_position", VehicleLocalPosition),
	"attitude": ("/fmu/out/vehicle_attitude", VehicleAttitude),
	"status": ("/fmu/out/vehicle_status", VehicleStatus),
}

MAX_EPH = 3.0  # max horizontal position error (meters) before recording is allowed

class StateRecorder(Node):
	def __init__(self):
		super().__init__("drone_state_recorder")
		self.latest = {}  # newest message of each topic
		# PX4 publishes best-effort, so the subscriber QoS must match
		qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
		                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
		                 history=HistoryPolicy.KEEP_LAST, depth=1)
		for name, (topic, msg_type) in TOPICS.items():
			self.create_subscription(msg_type, topic, lambda msg, name=name: self.save_msg(name, msg), qos)

	def save_msg(self, name, msg):
		self.latest[name] = msg

def main():
	# Start ROS2 and listen to the DDS agent in the background
	rclpy.init()
	node = StateRecorder()
	spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
	spin_thread.start()

	# Wait until every topic has data and the GPS position is accurate
	print("Waiting for GPS...")
	while True:
		pos = node.latest.get("global_position")
		if len(node.latest) == len(TOPICS) and pos.eph < MAX_EPH and not pos.dead_reckoning:
			break
		time.sleep(0.5)
	print("-- Drone data OK")

	# No filename: only the newest state, in the file the obstacle_*.py scripts read. Filename: every state
	keep_all = len(sys.argv) > 1
	filename = sys.argv[1] if keep_all else GPS_POINT_FILE
	if not os.path.splitext(filename)[1]:
		filename = filename + ".json"
	os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
	print(f"Saving to {filename}")

	states = []
	while(True):
		prompt = f"Enter to get drone state {len(states) + 1}" if keep_all else "Enter to save the current GPS point"
		text = input(f"{prompt}, enter q to quit: ")
		if (text == "q"):
			break
		# One state = time + the newest message of every topic
		state = {"time": datetime.now().isoformat()}
		for name, msg in node.latest.items():
			state[name] = message_to_ordereddict(msg)
		states = states + [state] if keep_all else [state]
		pos = node.latest["global_position"]
		print((pos.lat, pos.lon, pos.alt, pos.eph))
		with open(filename, 'w') as f:
			json.dump(states, f, indent=2, default=lambda x: x.item())
		print(f"Saved to {filename}")

	rclpy.shutdown()
	spin_thread.join(timeout=2.0)  # Python aborts at exit if this thread is still inside ROS
	node.destroy_node()

if __name__ == "__main__":
	main()