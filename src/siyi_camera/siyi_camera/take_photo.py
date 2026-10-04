#!/usr/bin/env python3
"""
siyi_photo.py
ROS2 services to move the gimbal and take photos with the SIYI camera (A8 mini).

    /siyi/set_gimbal          -> turn gimbal to (yaw, pitch) and wait until it gets there
    /siyi/take_photo_sd       -> photo saved on the camera SD card (UDP command)
    /siyi/take_photo_onboard  -> frame from RTSP video saved on the Jetson (1080p, fast)
    /siyi/take_photo_4k       -> photo on SD card, then downloaded to the Jetson (full 4K)

Run:
    ros2 run siyi_camera siyi_photo
    ros2 service call /siyi/set_gimbal siyi_interfaces/srv/SetGimbal "{yaw: 0.0, pitch: -90.0}"
    ros2 service call /siyi/take_photo_sd std_srvs/srv/Trigger
    ros2 service call /siyi/take_photo_onboard std_srvs/srv/Trigger
    ros2 service call /siyi/take_photo_4k std_srvs/srv/Trigger

Authors: Phuong Le (lephuo10@rowan.edu)
"""

import json
import os
import socket
import struct
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from siyi_interfaces.srv import SetGimbal
from std_srvs.srv import Trigger


def crc16(data):
    """CRC-16/XMODEM used by the SIYI protocol."""
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def make_packet(cmd_id, data, seq=0):
    """SIYI packet: 55 66 | CTRL | LEN(2) | SEQ(2) | CMD | DATA | CRC(2)"""
    body = b"\x55\x66" + struct.pack("<BHHB", 1, len(data), seq, cmd_id) + data
    return body + struct.pack("<H", crc16(body))


class SiyiPhoto(Node):
    def __init__(self):
        super().__init__("siyi_photo")
        self.ip = self.declare_parameter("ip", "192.168.144.25").value
        self.port = self.declare_parameter("port", 37260).value
        self.url = self.declare_parameter("rtsp_url", "rtsp://192.168.144.25:8554/main.264").value
        self.save_dir = os.path.expanduser(self.declare_parameter("save_dir", "~/px4_ros2_ws").value)
        os.makedirs(self.save_dir, exist_ok=True)

        # UDP socket for SIYI commands (SD card photo)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

        # Read the video in the background and keep only the newest frame (onboard photo)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self.read_video, daemon=True).start()

        self.create_service(SetGimbal, "/siyi/set_gimbal", self.set_gimbal)
        self.create_service(Trigger, "/siyi/take_photo_sd", self.take_photo_sd)
        self.create_service(Trigger, "/siyi/take_photo_onboard", self.take_photo_onboard)
        self.create_service(Trigger, "/siyi/take_photo_4k", self.take_photo_4k)
        self.get_logger().info(f"Ready: set_gimbal, take_photo_sd, take_photo_onboard, take_photo_4k -> {self.save_dir}")

    # ---------------- SD card ----------------

    def take_photo_sd(self, request, response):
        response.success, response.message = self.sd_photo()
        self.get_logger().info(response.message)
        return response

    def sd_photo(self):
        """Take a photo on the SD card. Returns (success, message)."""
        # 1. Ask camera info (0x0A): checks connection and SD card
        info = self.send(0x0A, b"", reply=0x0A)
        if info is None:
            return False, "Camera not answering (check IP / cable)"
        if len(info) >= 4 and info[3] == 2:  # record_sta 2 = no TF/SD card
            return False, "No SD card in camera"

        # 2. Take photo (0x0C, data 0x00)
        # Some firmware answers with 0x0B (0 = OK, 1 = failed), A8 mini may not
        result = self.send(0x0C, b"\x00", reply=0x0B, timeout=1.0)
        if result is not None and result[:1] == b"\x01":
            return False, "Camera failed to take photo"
        return True, "Photo saved to SD card"

    # ---------------- Gimbal ----------------

    def set_gimbal(self, request, response, timeout=5.0):
        # Keep the angle inside the A8 mini limits
        yaw = max(-135.0, min(135.0, request.yaw))
        pitch = max(-90.0, min(25.0, request.pitch))

        # 0x0E = set angle (yaw, pitch as int16 x10). The reply holds the current angle.
        data = struct.pack("<hh", round(yaw * 10), round(pitch * 10))
        response.success, response.message = False, f"Gimbal did not reach yaw={yaw}, pitch={pitch}"
        end = time.time() + timeout
        while time.time() < end:
            ack = self.send(0x0E, data, reply=0x0E, timeout=1.0)
            if ack is None:
                response.message = "Camera not answering (check IP / cable)"
            elif len(ack) >= 6:
                cur_yaw, cur_pitch, _ = struct.unpack("<hhh", ack[:6])
                response.yaw, response.pitch = cur_yaw / 10, cur_pitch / 10
                if abs(response.yaw - yaw) < 1.5 and abs(response.pitch - pitch) < 1.5:
                    time.sleep(0.5)  # let the gimbal settle (and the video catch up)
                    response.success, response.message = True, f"Gimbal at yaw={response.yaw}, pitch={response.pitch}"
                    break
            time.sleep(0.2)

        self.get_logger().info(response.message)
        return response

    def send(self, cmd_id, data, reply, timeout=2.0):
        """Send a command and wait for the reply packet (returns its data or None)."""
        self.clear_socket()
        self.seq = (self.seq + 1) % 65536
        self.sock.sendto(make_packet(cmd_id, data, self.seq), (self.ip, self.port))

        end = time.time() + timeout
        while time.time() < end:
            self.sock.settimeout(end - time.time())
            try:
                pkt, _ = self.sock.recvfrom(1024)
            except (socket.timeout, ValueError):
                return None
            if len(pkt) >= 10 and pkt[:2] == b"\x55\x66" and pkt[7] == reply:
                return pkt[8:-2]
        return None

    def clear_socket(self):
        """Drop old packets so we only read the answer to this request."""
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(1024)
        except (BlockingIOError, socket.error):
            pass

    # ---------------- Onboard (Jetson) ----------------

    def read_video(self):
        # Low delay: give us frames right away, do not buffer old ones
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        while rclpy.ok():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.get_logger().warn(f"Cannot open {self.url}, retrying...")
                time.sleep(2.0)
                continue

            self.get_logger().info("Video stream connected")
            while rclpy.ok():
                ok, frame = cap.read()
                if not ok:
                    self.get_logger().warn("Video stream lost, reconnecting...")
                    break
                with self.lock:
                    self.frame = frame
            cap.release()
            with self.lock:
                self.frame = None

    def take_photo_onboard(self, request, response):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()

        if frame is None:
            response.success, response.message = False, "No video from camera yet"
        else:
            name = datetime.now().strftime("photo_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
            path = os.path.join(self.save_dir, name)
            cv2.imwrite(path, frame)
            response.success, response.message = True, path

        self.get_logger().info(response.message)
        return response

    # ---------------- 4K: SD card + download ----------------

    def take_photo_4k(self, request, response):
        try:
            before = self.sd_files()
        except OSError as e:
            response.success, response.message = False, f"Camera web server not reachable: {e}"
            self.get_logger().info(response.message)
            return response

        ok, msg = self.sd_photo()
        if not ok:
            response.success, response.message = False, msg
            self.get_logger().info(msg)
            return response

        # Wait for the new file to show up on the SD card (camera needs ~1-2 s to write it)
        new = []
        end = time.time() + 10.0
        while not new and time.time() < end:
            time.sleep(0.5)
            try:
                new = [f for f in self.sd_files() if f["url"] not in before]
            except OSError:
                pass

        if not new:
            response.success, response.message = False, "Photo taken but new file not found on SD card"
        else:
            path = os.path.join(self.save_dir, new[-1]["name"])
            try:
                urllib.request.urlretrieve(new[-1]["url"], path)
                response.success, response.message = True, path
            except OSError as e:
                response.success, response.message = False, f"Download failed: {e}"

        self.get_logger().info(response.message)
        return response

    def media_api(self, api, **params):
        """Call the camera web server (port 82) and return its 'data' field."""
        query = urllib.parse.urlencode(params)
        url = f"http://{self.ip}:82//cgi-bin/media.cgi/api/v1/{api}?{query}"
        with urllib.request.urlopen(url, timeout=3.0) as r:
            return json.loads(r.read().decode()).get("data", {})

    def sd_files(self):
        """List photos in the newest folder on the SD card: [{'name', 'url'}, ...]"""
        dirs = self.media_api("getdirectories", media_type=0).get("directories", [])
        if not dirs:
            return []
        newest = max(dirs, key=lambda d: d["name"])["path"]
        count = self.media_api("getmediacount", media_type=0, path=newest).get("count", 0)
        files = self.media_api("getmedialist", media_type=0, path=newest, start=0, count=count)
        return files.get("list", [])


def main():
    rclpy.init()
    node = SiyiPhoto()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()