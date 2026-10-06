#!/usr/bin/env python3
"""
qr_stream.py
Authors: Phuong Le (lephuo10@rowan.edu)

Stream the SIYI camera (A8 mini) and detect QR codes live.
Three threads so the slow QR detection does not slow down the video:
    1. read_video  : read RTSP, keep only the newest frame
    2. detect_loop : find QR on the newest frame (small copy = faster)
    3. run (main)  : draw the last QR result on every frame and publish

    /siyi/qr_image/compressed  (sensor_msgs/CompressedImage) -> video with QR box
    /siyi/qr_data              (std_msgs/String)             -> QR text
    /siyi/qr_center            (geometry_msgs/Point)         -> QR offset from image center,
                                                                x right, y down, -0.5..0.5

    The video only shows a green box when a QR is read. QR text is printed on the Jetson terminal.

Run (Jetson):
    ros2 run siyi_camera qr_stream
    ros2 run siyi_camera qr_stream --ros-args -p detect:=false   # video only, to test max FPS
View (laptop, same ROS_DOMAIN_ID):
    ros2 run rqt_image_view rqt_image_view /siyi/qr_image/compressed
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class FPS:
    """Count frames per second."""
    def __init__(self):
        self.n, self.t, self.value = 0, time.time(), 0.0

    def tick(self):
        self.n += 1
        if time.time() - self.t >= 1.0:
            self.value, self.n, self.t = self.n / (time.time() - self.t), 0, time.time()


class QRStream(Node):
    def __init__(self):
        super().__init__("siyi_qr_stream")
        self.url = self.declare_parameter("rtsp_url", "rtsp://192.168.144.25:8554/main.264").value
        self.detect = self.declare_parameter("detect", True).value
        self.detect_width = self.declare_parameter("detect_width", 960).value  # smaller = faster, but QR must be bigger
        self.pub_width = self.declare_parameter("pub_width", 640).value        # size of the streamed image
        self.pub_rate = self.declare_parameter("pub_rate", 30.0).value         # max stream FPS
        self.quality = self.declare_parameter("jpeg_quality", 70).value

        self.img_pub = self.create_publisher(CompressedImage, "/siyi/qr_image/compressed", 1)
        self.data_pub = self.create_publisher(String, "/siyi/qr_data", 10)
        self.center_pub = self.create_publisher(Point, "/siyi/qr_center", 10)

        # Shared between threads
        self.lock = threading.Lock()
        self.frame, self.frame_id = None, 0
        self.qr = None  # last QR read: (corners as 0..1 of image size, time)
        self.running = True
        self.fps = FPS()

        self.threads = [threading.Thread(target=self.read_video, daemon=True)]
        if self.detect:
            self.threads.append(threading.Thread(target=self.detect_loop, daemon=True))
        for t in self.threads:
            t.start()
        self.get_logger().info("Streaming on /siyi/qr_image/compressed")

    # ---------------- 1. Camera ----------------

    def read_video(self):
        # Low delay: give us frames right away, do not buffer old ones
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        while self.running and rclpy.ok():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.get_logger().warn(f"Cannot open {self.url}, retrying...")
                time.sleep(2.0)
                continue

            self.get_logger().info("Video stream connected")
            while self.running and rclpy.ok():
                ok, frame = cap.read()
                if not ok:
                    self.get_logger().warn("Video stream lost, reconnecting...")
                    break
                with self.lock:
                    self.frame, self.frame_id = frame, self.frame_id + 1
            cap.release()

    def newest_frame(self, last_id):
        """Return (id, frame) if there is a frame newer than last_id, else (last_id, None)."""
        with self.lock:
            if self.frame is None or self.frame_id == last_id:
                return last_id, None
            return self.frame_id, self.frame

    # ---------------- 2. QR detection ----------------

    def detect_loop(self):
        detector = cv2.QRCodeDetector()
        last_id = -1
        while self.running and rclpy.ok():
            last_id, frame = self.newest_frame(last_id)
            if frame is None:
                time.sleep(0.002)
                continue

            h, w = frame.shape[:2]
            scale = min(1.0, self.detect_width / w)
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
            data, points, _ = detector.detectAndDecode(small)
            if not data:  # no QR, or QR seen but not readable
                continue

            # corners as 0..1 of the image size, so we can draw them on any image size
            corners = points.reshape(-1, 2) / [small.shape[1], small.shape[0]]
            with self.lock:
                self.qr = (corners, time.time())

            cx, cy = corners.mean(axis=0) - 0.5
            self.data_pub.publish(String(data=data))
            self.center_pub.publish(Point(x=float(cx), y=float(cy), z=0.0))
            self.get_logger().info(f"QR: '{data}'  offset x={cx:+.2f} y={cy:+.2f}",
                                   throttle_duration_sec=1.0)

    # ---------------- 3. Draw + stream ----------------

    def run(self):
        last_id, last_pub = -1, 0.0
        while rclpy.ok():
            last_id, frame = self.newest_frame(last_id)
            if frame is None:
                time.sleep(0.002)
                continue
            if time.time() - last_pub < 1.0 / self.pub_rate:
                continue
            last_pub = time.time()

            # Resize first, then draw on the small image (much cheaper than drawing on 1080p)
            h, w = frame.shape[:2]
            img = cv2.resize(frame, (self.pub_width, int(h * self.pub_width / w)))
            self.draw(img)
            self.publish(img)
            self.fps.tick()

        self.running = False

    def draw(self, img):
        h, w = img.shape[:2]
        with self.lock:
            qr = self.qr
        if qr and time.time() - qr[1] < 0.5:  # hide the box if no QR read for 0.5 s
            pts = (qr[0] * [w, h]).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        cv2.putText(img, f"FPS {self.fps.value:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def publish(self, img):
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "siyi_camera"
        msg.format = "jpeg"
        msg.data = jpg.tobytes()
        self.img_pub.publish(msg)


def main():
    rclpy.init()
    node = QRStream()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.running = False
    for t in node.threads:
        t.join(timeout=3.0)  # let the threads close the stream cleanly
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()