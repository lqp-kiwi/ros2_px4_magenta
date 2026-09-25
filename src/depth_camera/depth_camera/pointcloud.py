#!/usr/bin/env python3
"""
pointcloud_filter_node

Nhan PointCloud2 tu driver Orbbec (mac dinh /camera/depth/points),
loc theo khoang cach + voxel downsample, roi publish ra topic moi
(mac dinh /obstacle_points) nhe hon nhieu de xem tren RViz2,
ke ca khi RViz chay tren laptop va nhan du lieu qua WiFi.

Tat ca tham so doc lai o moi frame, nen co the chinh truc tiep khi dang chay:
    ros2 param set /pointcloud_filter_node voxel_size 0.1
"""
import array
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import PointCloud2, PointField


# ---------------------------------------------------------------------------
# Ham tien ich: chuyen doi PointCloud2 <-> numpy (khong phu thuoc PCL)
# ---------------------------------------------------------------------------
def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Tra ve mang (N, 3) float32 chua x, y, z; da bo cac diem NaN/inf."""
    fields = {f.name: f for f in msg.fields}
    for name in ('x', 'y', 'z'):
        if name not in fields:
            raise ValueError(f'PointCloud2 khong co truong "{name}"')
        if fields[name].datatype != PointField.FLOAT32:
            raise ValueError(f'Truong "{name}" khong phai FLOAT32')

    n_points = msg.width * msg.height
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [endian + 'f4'] * 3,
        'offsets': [fields['x'].offset, fields['y'].offset, fields['z'].offset],
        'itemsize': msg.point_step,
    })

    if msg.row_step == msg.width * msg.point_step:
        raw = np.frombuffer(msg.data, dtype=dtype, count=n_points)
    else:
        # Truong hop moi hang co byte dem (padding) o cuoi
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        buf = buf.reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
        raw = np.ascontiguousarray(buf).view(dtype).reshape(-1)

    xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32, copy=False)
    return xyz[np.isfinite(xyz).all(axis=1)]


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Giu lai 1 diem cho moi o voxel kich thuoc voxel_size (m)."""
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, idx = np.unique(flat, return_index=True)
    return xyz[idx]


def xyz_to_pointcloud2(xyz: np.ndarray, header) -> PointCloud2:
    """Dong goi mang (N, 3) thanh PointCloud2 khong co cau truc (height = 1)."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(('x', 'y', 'z'))
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = True
    msg.data = array.array('B', xyz.astype('<f4').tobytes())
    return msg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class PointCloudFilterNode(Node):

    def __init__(self):
        super().__init__('pointcloud_filter_node')

        self.declare_parameter('input_topic', '/camera/depth/points')
        self.declare_parameter('output_topic', '/obstacle_points')
        self.declare_parameter('min_range', 0.2)    # m; cung loai bo diem (0,0,0) khong hop le
        self.declare_parameter('max_range', 5.0)    # m
        self.declare_parameter('voxel_size', 0.05)  # m; <= 0 de tat downsample
        self.declare_parameter('max_rate_hz', 10.0)  # <= 0 de khong gioi han

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # Sensor QoS (best effort) tuong thich voi ca publisher reliable lan best effort
        self.sub = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, qos_profile_sensor_data)

        # Reliable de RViz / node khac subscribe voi QoS nao cung duoc
        out_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, out_qos)

        self.last_pub_time = 0.0
        self.get_logger().info(f'Nhan tu "{input_topic}" -> publish ra "{output_topic}"')

    def cloud_callback(self, msg: PointCloud2):
        now = time.monotonic()
        max_rate = float(self.get_parameter('max_rate_hz').value)
        if max_rate > 0.0 and (now - self.last_pub_time) < 1.0 / max_rate:
            return

        min_range = float(self.get_parameter('min_range').value)
        max_range = float(self.get_parameter('max_range').value)
        voxel_size = float(self.get_parameter('voxel_size').value)

        try:
            xyz = pointcloud2_to_xyz(msg)
        except ValueError as err:
            self.get_logger().error(str(err), throttle_duration_sec=5.0)
            return
        n_in = xyz.shape[0]

        # 1) Loc theo khoang cach tu camera
        dist = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(dist >= min_range) & (dist <= max_range)]

        # 2) Voxel downsample
        if voxel_size > 0.0 and xyz.shape[0] > 0:
            xyz = voxel_downsample(xyz, voxel_size)

        # Giu nguyen header -> cung frame_id va timestamp voi cloud goc
        self.pub.publish(xyz_to_pointcloud2(xyz, msg.header))
        self.last_pub_time = now

        dt_ms = (time.monotonic() - now) * 1000.0
        self.get_logger().info(
            f'{n_in} -> {xyz.shape[0]} diem | xu ly {dt_ms:.1f} ms | frame "{msg.header.frame_id}"',
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()