#!/usr/bin/env python3
"""
    Simulation only: flies a box at the drone, to test px4_control/obstacle_avoidance_test.py.
    The box appears in front of the drone (camera direction) at the height of the drone, moves straight
    back along the camera direction, past the drone, and is removed. With --speed 0 it just stays there
    for --duration seconds. At the end it prints how close it came and how far the drone moved.
    Gazebo and PX4 must be running.

    Usage:
        python3 moving_obstacle.py [center|left|right] [--speed M_S] [--start M] [--size M] [--offset M]
        center: straight at the drone, left / right: --offset meters to that side of it
"""

import argparse
import math
import re
import sys
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

DRONE = 'x500_depth_down_0'
BOX = 'obstacle'
DRONE_RADIUS = 0.38     # m, center of the x500 to the tip of a propeller
PAST_DRONE = 3.0        # m, the box is removed this far behind the drone
SERVICE_TIMEOUT = 500   # ms. The answer of a Gazebo service is sometimes lost even though the call worked,
                        # so it is ignored: the box showing up in (or leaving) the poses is what counts
BOX_TIMEOUT = 3.0       # s to wait for the box to show up / be gone

# No gravity and a velocity command: the box flies straight, and still pushes the drone away if it hits it
BOX_SDF = """<sdf version="1.9">
  <model name="{name}">
    <pose>{x} {y} {z} 0 0 {yaw}</pose>
    <link name="link">
      <gravity>false</gravity>
      <inertial>
        <mass>5</mass>
        <inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{s} {s} {s}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{s} {s} {s}</size></box></geometry>
        <material><ambient>0.8 0.1 0.1 1</ambient><diffuse>0.8 0.1 0.1 1</diffuse></material>
      </visual>
    </link>
    <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"/>
  </model>
</sdf>"""


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def surface_distance(box, drone, half):
    """Distance from the center of the drone to the surface of the box (m)"""
    c, s = math.cos(box[3]), math.sin(box[3])
    dx, dy, dz = drone[0] - box[0], drone[1] - box[1], drone[2] - box[2]
    return math.sqrt(max(abs(c * dx + s * dy) - half, 0) ** 2 + max(abs(-s * dx + c * dy) - half, 0) ** 2
                     + max(abs(dz) - half, 0) ** 2)


def wait_until(condition, timeout):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def main():
    parser = argparse.ArgumentParser(description='Fly a box at the drone (simulation only)')
    parser.add_argument('side', nargs='?', default='center', choices=['center', 'left', 'right'])
    parser.add_argument('--speed', type=float, default=1.0, help='m/s, 0 = stays where it appears')
    parser.add_argument('--start', type=float, default=7.0, help='m in front of the drone')
    parser.add_argument('--size', type=float, default=1.0, help='m, side of the box')
    parser.add_argument('--offset', type=float, default=1.0, help='m to the side for left / right')
    parser.add_argument('--duration', type=float, default=10.0, help='s, how long the box stays with --speed 0')
    args = parser.parse_args()

    node = Node()
    worlds = [m.group(1) for m in map(re.compile(r'/world/([^/]+)/clock').fullmatch, node.topic_list()) if m]
    if not worlds:
        print('Gazebo is not running')
        return 1
    world = worlds[0]

    poses = {}

    def pose_cb(msg):
        poses.clear()
        for p in msg.pose:
            if p.name in (DRONE, BOX):
                poses[p.name] = (p.position.x, p.position.y, p.position.z, yaw_of(p.orientation))

    node.subscribe(Pose_V, f'/world/{world}/dynamic_pose/info', pose_cb)
    if not wait_until(lambda: DRONE in poses, 5.0):
        print(f'No pose of {DRONE}: is PX4 running?')
        return 1

    def remove_box():
        if BOX in poses:
            node.request(f'/world/{world}/remove', Entity(name=BOX, type=Entity.MODEL), Entity, Boolean,
                         SERVICE_TIMEOUT)
            wait_until(lambda: DRONE in poses and BOX not in poses, BOX_TIMEOUT)

    remove_box()    # left over from an interrupted run

    # Box in front of the drone, left = +y in the frame of the drone (Gazebo: x forward, y left)
    x0, y0, z0, yaw = poses[DRONE]
    lateral = {'center': 0.0, 'left': args.offset, 'right': -args.offset}[args.side]
    sdf = BOX_SDF.format(name=BOX, s=args.size, yaw=yaw, z=z0,
                         x=x0 + args.start * math.cos(yaw) - lateral * math.sin(yaw),
                         y=y0 + args.start * math.sin(yaw) + lateral * math.cos(yaw))
    node.request(f'/world/{world}/create', EntityFactory(sdf=sdf), EntityFactory, Boolean, SERVICE_TIMEOUT)
    if not wait_until(lambda: BOX in poses, BOX_TIMEOUT):
        print('Could not create the box')
        return 1

    cmd_pub = node.advertise(f'/model/{BOX}/cmd_vel', Twist)
    cmd = Twist()
    cmd.linear.x = -args.speed      # in the frame of the box, which faces the same way as the drone
    print(f'Box {args.size} m, {args.start} m ahead, {args.side}, {args.speed} m/s')

    half = args.size / 2
    travel = args.start + PAST_DRONE
    closest = math.inf
    start_time = time.monotonic()
    box_start = poses[BOX]
    try:
        while True:
            cmd_pub.publish(cmd)
            time.sleep(0.05)
            box, drone = poses.get(BOX), poses.get(DRONE)
            if box is None or drone is None:    # pose_cb is just filling them in
                continue
            closest = min(closest, surface_distance(box, drone, half))
            moved = math.hypot(box[0] - box_start[0], box[1] - box_start[1])
            if moved > travel if args.speed > 0 else time.monotonic() - start_time > args.duration:
                break
    finally:
        cmd_pub.publish(Twist())
        remove_box()

    x1, y1, z1, _ = drone
    right = (x1 - x0) * math.sin(yaw) - (y1 - y0) * math.cos(yaw)
    forward = (x1 - x0) * math.cos(yaw) + (y1 - y0) * math.sin(yaw)
    print(f'Closest: {closest:.2f} m from the center of the drone to the box '
          f'({"HIT" if closest < DRONE_RADIUS else "no contact"}, propeller tips at {DRONE_RADIUS} m)')
    print(f'Drone moved: {right:+.2f} m to the right, {forward:+.2f} m forward, {z1 - z0:+.2f} m up')
    return 0


if __name__ == '__main__':
    sys.exit(main())
