#!/usr/bin/env python3

import sys
import json
import time
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from px4_msgs.msg import VehicleGlobalPosition, VehicleLocalPosition, VehicleAttitude, VehicleStatus

"""
Script Name: record_drone_state.py
Description: This script creates a .json file of the drone's current state,
             saving the raw px4_msgs messages returned by the uXRCE-DDS agent

Last Modified: 2026-10-03

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

filename = ""

def filename_valid(filename: str) -> bool:
	dot_index = filename.find(".")
	return dot_index == -1 or filename[dot_index:] == ".json"

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
	threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

	# Wait until every topic has data and the GPS position is accurate
	print("Waiting for GPS...")
	while True:
		pos = node.latest.get("global_position")
		if len(node.latest) == len(TOPICS) and pos.eph < MAX_EPH and not pos.dead_reckoning:
			break
		time.sleep(0.5)
	print("-- Drone data OK")

	global filename
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		print(f"Generating new .json file: {filename}")
	else:
		while True:
			filename = input("Enter the name of the file you want to save to. To not save, enter 'n' (case sensitive):")
			if filename_valid(filename):
				break
			else:
				print("This filename is invalid, because it contains a period, without ending in '.json'.")
				print("Enter a name without a period, or with the above extension.")
	do_save = filename != "n"
	if filename.find(".") == -1:
		filename = filename + ".json"

	states = []
	while(True):
		text = input(f"Enter to get drone state {len(states) + 1}, enter q to quit")
		if (text == "q"):
			break
		# One state = time + the newest message of every topic
		state = {"time": datetime.now().isoformat()}
		for name, msg in node.latest.items():
			state[name] = message_to_ordereddict(msg)
		states.append(state)
		pos = node.latest["global_position"]
		print((pos.lat, pos.lon, pos.alt, pos.eph))
		if do_save:
			with open(filename, 'w') as f:
				json.dump(states, f, indent=2, default=lambda x: x.item())

	node.destroy_node()
	rclpy.shutdown()

if __name__ == "__main__":
	main()