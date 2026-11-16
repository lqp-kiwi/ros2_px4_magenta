#!/usr/bin/env python3
"""
qr_detect.py
Authors: Phuong Le (lephuo10@rowan.edu)

Connect to the SIYI camera (A8 mini) RTSP video, look for a QR code,
save the image when found, then stop.

    <save_dir>/qr_<time>.jpg      -> raw frame
    <save_dir>/qr_<time>_box.jpg  -> frame with QR box

Run:
    ros2 run siyi_camera qr_detect
    ros2 run siyi_camera qr_detect --ros-args -p timeout:=60.0
"""

import os
import threading
import time
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node


class QRDetect(Node):
    def __init__(self):
        super().__init__("siyi_qr_detect")
        self.url = self.declare_parameter("rtsp_url", "rtsp://192.168.144.25:8554/main.264").value
        self.save_dir = os.path.expanduser(self.declare_parameter("save_dir", "~/px4_ros2_ws/data/2026/qr_images").value)
        self.timeout = self.declare_parameter("timeout", 0.0).value  # seconds, 0 = wait forever
        os.makedirs(self.save_dir, exist_ok=True)

        self.detector = cv2.QRCodeDetector()

        # Read the video in the background and keep only the newest frame
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.read_video, daemon=True)
        self.thread.start()

    # ---------------- Camera ----------------

    def read_video(self):
        # Low delay: give us frames right away, do not buffer old ones
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        while self.running and rclpy.ok():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.get_logger().warn(f"Cannot open {self.url}, retrying...")
                time.sleep(2.0)
                continue

            self.get_logger().info("Video stream connected, looking for QR code...")
            while self.running and rclpy.ok():
                ok, frame = cap.read()
                if not ok:
                    self.get_logger().warn("Video stream lost, reconnecting...")
                    break
                with self.lock:
                    self.frame = frame
            cap.release()
            with self.lock:
                self.frame = None

    # ---------------- QR ----------------

    def run(self):
        """Check the newest frame until a QR code is found (or timeout)."""
        start = time.time()
        while rclpy.ok():
            if self.timeout > 0 and time.time() - start > self.timeout:
                self.get_logger().warn(f"No QR code found after {self.timeout} s")
                break

            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
                self.frame = None  # do not check the same frame twice
            if frame is None:
                time.sleep(0.01)
                continue

            data, points, _ = self.detector.detectAndDecode(frame)
            if data:  # empty string = no QR, or QR seen but not readable
                self.save(frame, data, points)
                break

        self.running = False  # stop the camera thread

    def save(self, frame, data, points):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(self.save_dir, f"qr_{stamp}.jpg")
        cv2.imwrite(raw_path, frame)

        # Draw the QR box and its text
        pts = points.reshape(-1, 2).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
        box_path = os.path.join(self.save_dir, f"qr_{stamp}_box.jpg")
        cv2.imwrite(box_path, frame)

        self.get_logger().info(f"QR found: '{data}'")
        self.get_logger().info(f"Saved: {raw_path}")
        self.get_logger().info(f"Saved: {box_path}")


def main():
    rclpy.init()
    node = QRDetect()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.running = False
    node.thread.join(timeout=3.0)  # let the camera thread close the stream cleanly
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()