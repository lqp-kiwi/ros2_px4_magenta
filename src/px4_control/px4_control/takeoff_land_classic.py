#!/usr/bin/env python3

"""
    Takeoff and Land for PX4 using ROS2.
    Author: Phuong Le
    Lasted Updated: 2026-09-19
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand

class TakeoffLandNode(Node):
    def __init__(self):
        super().__init__('takeoff_land_node')

        # Config QoS for PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create timer (duration 1 second)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.seconds_passed = 0
        
        self.target_system = 10

    def timer_callback(self):
        self.seconds_passed += 1

        if self.seconds_passed == 1:
            self.arm()
        elif self.seconds_passed == 5:
            # Takeoff command after 1 second of arming
            self.takeoff(altitude=5.0)
        elif self.seconds_passed == 20:
            # After 13 seconds of flying at a constant altitude, send the Land command
            self.land()
        elif self.seconds_passed == 30:
            # Lock the drone after 10 seconds of landing
            self.disarm()
        elif self.seconds_passed > 40:
            self.get_logger().info(">>> Completed Takeoff and Landing. Shutting down Node...")
            raise SystemExit 

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f">>> Sent ARM to Vehicle {self.target_system} <<<")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info(f">>> Sent DISARM to Vehicle {self.target_system} <<<")

    def takeoff(self, altitude=1.5):
        # Param7 in NAV_TAKEOFF: attidue to take off (meters)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=float(altitude))
        self.get_logger().info(f">>> Sent TAKEOFF command (altitude: {altitude}m) to Vehicle {self.target_system} <<<")

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info(f">>> Sent LAND command to Vehicle {self.target_system} <<<")

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
    node = TakeoffLandNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()