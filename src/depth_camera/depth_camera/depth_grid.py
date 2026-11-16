"""Split the Orbbec depth image into an n x n grid and publish a visualization image.

Usage:
  ros2 run depth_camera camera --no-cloud           # run the camera first
  ros2 run depth_camera depth_grid                  # grid 5x5
  ros2 run depth_camera depth_grid -p grid_n:=8     # override any param

View (on your laptop, same network + same ROS_DOMAIN_ID):
  ros2 run rqt_image_view rqt_image_view /depth_grid/image
  or rviz2 -> Add -> By topic -> /depth_grid/image -> Image
  Over WiFi pick the "compressed" transport (much lighter than raw).

Topics:
  in : /camera/depth/image_raw          depth image (Y16 -> 16UC1, millimeters)
  out: /depth_grid                      Float32MultiArray, n x n mean distance (m), row-major, NaN = no data
  out: /depth_grid/image                visualization image (bgr8)
  out: /depth_grid/image/compressed     same image as JPEG (for WiFi)
"""
import sys
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge

# ---------------- Config (edit here) ----------------
PARAMS = {
    'depth_topic': '/camera/depth/image_raw',
    'grid_n': 5,              # grid n x n
    'max_range': 5.0,         # meters; same as threshold_filter_max (5000 mm) in camera.py
    'image_fps': 15,          # max FPS of /depth_grid/image (lower = lighter on WiFi)
}
DISPLAY_WIDTH = 640           # width of the visualization image (px)
JPEG_QUALITY = 80
# ----------------------------------------------------


# ---------------------------------------------------------------- processing
def to_meters(img, encoding, max_range):
    """Convert raw depth image to float meters. NaN = no data (0 in Y16)."""
    if encoding in ('16UC1', 'mono16'):
        depth = img.astype(np.float32) / 1000.0   # millimeters -> meters
    elif encoding == '32FC1':
        depth = img.astype(np.float32)
        depth[np.isposinf(depth)] = max_range      # out of range = free space
    else:
        return None

    depth[~(depth > 0.0)] = np.nan                 # 0 / negative / NaN = no data
    return np.minimum(depth, max_range)            # farther than max_range = max_range


def compute_grid(depth, n):
    """Mean distance of each cell in an n x n grid. NaN if the cell has no valid pixel."""
    h, w = depth.shape
    ch, cw = h // n, w // n
    d = depth[:ch * n, :cw * n].reshape(n, ch, n, cw)   # (row, pix_y, col, pix_x)

    valid = np.isfinite(d)
    sums = np.where(valid, d, 0.0).sum(axis=(1, 3))     # (n, n)
    counts = valid.sum(axis=(1, 3))                     # (n, n)

    grid = np.full((n, n), np.nan)
    np.divide(sums, counts, out=grid, where=counts > 0)
    return grid


# ---------------------------------------------------------------- drawing
def put_label(img, text, org, scale, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(depth, grid, max_range, fps=None):
    """Depth image (near = red, far = blue, no data = black) + grid + mean distance per cell."""
    n = grid.shape[0]
    h, w = depth.shape
    depth = depth[:(h // n) * n, :(w // n) * n]          # same crop as compute_grid
    disp_w = DISPLAY_WIDTH
    disp_h = int(depth.shape[0] * disp_w / depth.shape[1])

    norm = np.nan_to_num(depth, nan=max_range) / max_range
    view = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view[~np.isfinite(depth)] = (0, 0, 0)
    view = cv2.resize(view, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    cell_w, cell_h = disp_w / n, disp_h / n
    for i in range(1, n):
        cv2.line(view, (int(i * cell_w), 0), (int(i * cell_w), disp_h), (255, 255, 255), 1)
        cv2.line(view, (0, int(i * cell_h)), (disp_w, int(i * cell_h)), (255, 255, 255), 1)

    scale = max(0.3, min(cell_w, cell_h) / 110.0)
    for r in range(n):
        for c in range(n):
            text = '--' if np.isnan(grid[r, c]) else f'{grid[r, c]:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            org = (int((c + 0.5) * cell_w - tw / 2), int((r + 0.5) * cell_h + th / 2))
            put_label(view, text, org, scale)

    if fps is not None:
        put_label(view, f'{fps:.1f} FPS', (disp_w - 90, 18), 0.5, (0, 255, 0))
    return view


# ---------------------------------------------------------------- ROS2 node
class DepthGridNode(Node):
    def __init__(self):
        super().__init__('depth_grid')

        for name, value in PARAMS.items():
            self.declare_parameter(name, value)
        self.depth_topic = self.get_parameter('depth_topic').value
        self.n = int(self.get_parameter('grid_n').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.image_period = 1.0 / max(float(self.get_parameter('image_fps').value), 1.0)

        # Input: sensor topic -> Best Effort, keep only the newest frame
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Output image: Reliable works with both rqt_image_view and rviz2 (Reliable or Best Effort)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.bridge = CvBridge()
        self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
        self.grid_pub = self.create_publisher(Float32MultiArray, '/depth_grid', 10)
        self.image_pub = self.create_publisher(Image, '/depth_grid/image', image_qos)
        self.compressed_pub = self.create_publisher(
            CompressedImage, '/depth_grid/image/compressed', image_qos)
        self.last_image_time = 0.0

        # FPS + watchdog: warn if no depth frame arrives
        self.last_frame_time = None
        self.fps = None
        self.create_timer(3.0, self.watchdog_callback)

        self.get_logger().info(
            f">>> Listening {self.depth_topic} | grid {self.n}x{self.n} | max_range {self.max_range} m <<<")
        self.get_logger().info(">>> View: rqt_image_view /depth_grid/image  (or rviz2 Image display) <<<")

    def depth_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = to_meters(img, msg.encoding, self.max_range)
        if depth is None:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return

        grid = compute_grid(depth, self.n)
        self.publish_grid(grid)
        self.update_fps()

        # Terminal output (1 time / second) so you always know it is running
        self.get_logger().info(
            f"[{self.fps or 0:.1f} FPS] mean distance grid (m):\n"
            + np.array2string(grid, precision=2, suppress_small=True),
            throttle_duration_sec=1.0)

        self.publish_image(depth, grid, msg.header)

    def publish_image(self, depth, grid, header):
        # Only draw when someone is watching, and at most image_fps
        want_raw = self.image_pub.get_subscription_count() > 0
        want_jpeg = self.compressed_pub.get_subscription_count() > 0
        now = time.monotonic()
        if not (want_raw or want_jpeg) or now - self.last_image_time < self.image_period:
            return
        self.last_image_time = now

        vis = render(depth, grid, self.max_range, self.fps)
        if want_raw:
            out = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            out.header = header
            self.image_pub.publish(out)
        if want_jpeg:
            ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                out = CompressedImage()
                out.header = header
                out.format = 'jpeg'
                out.data = buf.tobytes()
                self.compressed_pub.publish(out)

    def update_fps(self):
        now = time.monotonic()
        if self.last_frame_time is not None:
            inst = 1.0 / max(now - self.last_frame_time, 1e-6)
            self.fps = inst if self.fps is None else 0.9 * self.fps + 0.1 * inst
        self.last_frame_time = now

    def watchdog_callback(self):
        if self.last_frame_time is None or time.monotonic() - self.last_frame_time > 3.0:
            self.get_logger().warn(
                f"No depth frame on {self.depth_topic}. Is 'ros2 run depth_camera camera' running?")

    def publish_grid(self, grid):
        n = self.n
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='rows', size=n, stride=n * n),
            MultiArrayDimension(label='cols', size=n, stride=n),
        ]
        msg.data = [float(v) for v in grid.flatten()]
        self.grid_pub.publish(msg)


def main():
    args = sys.argv[1:]
    # Allow "-p name:=value" without typing --ros-args (same as camera.py)
    if args and '--ros-args' not in args:
        args = ['--ros-args'] + args
    rclpy.init(args=[sys.argv[0]] + args)
    node = DepthGridNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()