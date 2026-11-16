#!/usr/bin/env python3
"""
record_video.py
Authors: Phuong Le (lephuo10@rowan.edu)

ROS2 services to start / stop recording video with the SIYI camera (A8 mini).

    /siyi/start_record  -> start recording
    /siyi/stop_record   -> stop recording and save the video

Where the video is saved (parameter save_to):
    onboard : RTSP video copied to an .mp4 file on the Jetson (full quality, almost no CPU)
    sd      : camera records to its own SD card (UDP command)
    both    : both at the same time

Run:
    ros2 run siyi_camera record_video
    ros2 run siyi_camera record_video --ros-args -p save_to:=sd
    ros2 service call /siyi/start_record std_srvs/srv/Trigger
    ros2 service call /siyi/stop_record std_srvs/srv/Trigger

Onboard recording needs ffmpeg:  sudo apt install ffmpeg
"""

import os
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from siyi_camera.siyi_sdk import build_packet

# record_sta in the camera info reply (0x0A)
NOT_RECORDING, RECORDING, NO_SD_CARD = 0, 1, 2
DATA_DIR = "~/px4_ros2_ws/data/2026/videos/"


class SiyiRecord(Node):
    def __init__(self):
        super().__init__("siyi_record")
        self.ip = self.declare_parameter("ip", "192.168.144.25").value
        self.port = self.declare_parameter("port", 37260).value
        self.url = self.declare_parameter("rtsp_url", "rtsp://192.168.144.25:8554/main.264").value
        self.save_dir = os.path.expanduser(self.declare_parameter("save_dir", DATA_DIR).value)
        self.save_to = self.declare_parameter("save_to", "onboard").value  # onboard | sd | both
        os.makedirs(self.save_dir, exist_ok=True)

        # UDP socket for SIYI commands (SD card recording)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

        # Onboard recording (ffmpeg process)
        self.proc = None
        self.path = None
        self.start_time = 0.0

        self.create_service(Trigger, "/siyi/start_record", self.start_record)
        self.create_service(Trigger, "/siyi/stop_record", self.stop_record)
        self.get_logger().info(f"Ready: start_record, stop_record (save_to={self.save_to}) -> {self.save_dir}")

    # ---------------- Services ----------------

    def start_record(self, request, response):
        results = []
        if self.save_to in ("onboard", "both"):
            results.append(self.onboard_start())
        if self.save_to in ("sd", "both"):
            results.append(self.sd_record(start=True))
        return self.answer(response, results)

    def stop_record(self, request, response):
        results = []
        if self.save_to in ("onboard", "both"):
            results.append(self.onboard_stop())
        if self.save_to in ("sd", "both"):
            results.append(self.sd_record(start=False))
        return self.answer(response, results)

    def answer(self, response, results):
        response.success = all(ok for ok, _ in results)
        response.message = " | ".join(msg for _, msg in results)
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().warn(response.message)
        return response

    # ---------------- Onboard (Jetson) ----------------

    def onboard_start(self):
        if self.proc is not None and self.proc.poll() is None:
            return True, f"Onboard: already recording {self.path}"
        if shutil.which("ffmpeg") is None:
            return False, "Onboard: ffmpeg not found (sudo apt install ffmpeg)"

        self.path = os.path.join(self.save_dir, datetime.now().strftime("video_%Y%m%d_%H%M%S.mp4"))
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-i", self.url,
            "-c", "copy",  # no re-encoding: keep camera quality, almost no CPU
            # fragmented mp4: the file is still playable if the Jetson loses power
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-y", self.path,
        ]
        self.log = open(os.path.join(self.save_dir, "ffmpeg.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self.log)
        self.start_time = time.time()

        # Wait until ffmpeg is connected and writing the file
        end = time.time() + 10.0
        while time.time() < end:
            if self.proc.poll() is not None:
                self.onboard_cleanup()
                return False, f"Onboard: ffmpeg stopped, see {self.log.name}"
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                return True, f"Onboard: recording -> {self.path}"
            time.sleep(0.1)

        self.onboard_cleanup()
        return False, "Onboard: no video from camera (check IP / cable)"

    def onboard_cleanup(self):
        """Start failed: stop ffmpeg and remove the empty file."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        self.log.close()
        if os.path.exists(self.path) and os.path.getsize(self.path) == 0:
            os.remove(self.path)

    def onboard_stop(self):
        if self.proc is None:
            return False, "Onboard: not recording"

        # Ctrl+C = ffmpeg closes the file properly
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.log.close()

        duration = time.time() - self.start_time
        size = os.path.getsize(self.path) / 1e6 if os.path.exists(self.path) else 0.0
        if size < 0.01:
            return False, f"Onboard: video is empty, see {self.log.name}"
        return True, f"Onboard: saved {self.path} ({duration:.0f} s, {size:.1f} MB)"

    # ---------------- SD card ----------------

    def sd_record(self, start):
        """Start (start=True) or stop (start=False) recording on the SD card."""
        want = RECORDING if start else NOT_RECORDING
        word = "recording" if start else "stopped"

        # 1. Ask camera info (0x0A): checks connection, SD card and record state
        state = self.record_state()
        if state is None:
            return False, "SD: camera not answering (check IP / cable)"
        if state == NO_SD_CARD:
            return False, "SD: no SD card in camera"
        if state == want:
            return True, f"SD: already {word}"

        # 2. Toggle recording (0x0C, data 0x02). Same command for start and stop.
        self.send(0x0C, b"\x02", reply=0x0B, timeout=1.0)

        # 3. Check that the state really changed
        end = time.time() + 3.0
        while time.time() < end:
            if self.record_state() == want:
                return True, f"SD: {word}"
            time.sleep(0.3)
        return False, f"SD: camera did not switch to {word}"

    def record_state(self):
        info = self.send(0x0A, b"", reply=0x0A)
        if info is None or len(info) < 4:
            return None
        return info[3]

    def send(self, cmd_id, data, reply, timeout=2.0):
        """Send a command and wait for the reply packet (returns its data or None)."""
        self.clear_socket()
        self.seq = (self.seq + 1) % 65536
        self.sock.sendto(build_packet(cmd_id, data, self.seq), (self.ip, self.port))

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


def main():
    rclpy.init()
    node = SiyiRecord()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.proc is not None:  # do not lose the video if the node is stopped while recording
        node.get_logger().info(node.onboard_stop()[1])
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()