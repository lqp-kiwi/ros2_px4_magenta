#!/usr/bin/env python3
"""Arm or disarm the PX4 vehicle, then exit.

Usage: ros2 run px4_arm arm_node [arm|disarm]
"""

import sys

import rclpy
from rclpy.node import Node

from px4_msgs.msg import VehicleCommand


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'arm'
    if action not in ('arm', 'disarm'):
        print(f"unknown action '{action}', expected 'arm' or 'disarm'")
        return 1

    rclpy.init()
    node = Node('arm_node')
    pub = node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 1)

    # Wait for the uXRCE-DDS bridge, otherwise the command goes nowhere.
    while pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)

    msg = VehicleCommand()
    msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    msg.param1 = 1.0 if action == 'arm' else 0.0
    msg.target_system = 1
    msg.target_component = 1
    pub.publish(msg)

    node.get_logger().info(action.upper())
    rclpy.spin_once(node, timeout_sec=0.5)
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
