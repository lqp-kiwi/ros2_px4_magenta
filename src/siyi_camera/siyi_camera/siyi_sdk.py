"""
Minimal SIYI gimbal SDK (UDP) for A8 mini.

Packet format (little-endian):
  STX(2)=0x55 0x66 | CTRL(1) | DATA_LEN(2) | SEQ(2) | CMD_ID(1) | DATA(n) | CRC16(2)
CRC16: CCITT/XMODEM (poly 0x1021, init 0) over every byte before the CRC.
Default camera IP 192.168.144.25, UDP port 37260.
"""
import socket
import struct
import threading


# ---- Command IDs (SIYI external SDK) ----
CMD_FIRMWARE = 0x01
CMD_MANUAL_ZOOM = 0x05      # int8: 1 zoom in, 0 stop, -1 zoom out
CMD_GIMBAL_SPEED = 0x07     # int8 yaw, int8 pitch  (-100..100)
CMD_CENTER = 0x08           # uint8 1
CMD_FUNCTION = 0x0C         # uint8: 0 photo, 2 record toggle, 3 lock, 4 follow, 5 FPV
CMD_GET_ATTITUDE = 0x0D     # reply: int16 yaw,pitch,roll,yaw_v,pitch_v,roll_v (x10)
CMD_SET_ANGLE = 0x0E        # int16 yaw, int16 pitch (deg x10)
CMD_ABS_ZOOM = 0x0F         # uint8 integer part, uint8 decimal part

FUNC_PHOTO = 0
FUNC_RECORD = 2
FUNC_LOCK = 3
FUNC_FOLLOW = 4
FUNC_FPV = 5


def crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def build_packet(cmd_id: int, data: bytes = b"", seq: int = 0, ctrl: int = 0x01) -> bytes:
    head = struct.pack("<BBBHHB", 0x55, 0x66, ctrl, len(data), seq & 0xFFFF, cmd_id)
    body = head + data
    return body + struct.pack("<H", crc16(body))


def parse_packets(buf: bytes):
    """Yield (cmd_id, data) for each valid packet in a UDP datagram
    (one datagram can occasionally hold more than one packet)."""
    i = 0
    while i + 10 <= len(buf):
        if buf[i] != 0x55 or buf[i + 1] != 0x66:
            i += 1
            continue
        _, _, _, dlen, _, cmd = struct.unpack_from("<BBBHHB", buf, i)
        end = i + 8 + dlen
        if end + 2 > len(buf):
            break
        (crc_rx,) = struct.unpack_from("<H", buf, end)
        if crc16(buf[i:end]) == crc_rx:
            yield cmd, buf[i + 8:end]
        i = end + 2


class SiyiGimbal:
    def __init__(self, ip="192.168.144.25", port=37260, timeout=0.5):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self._seq = 0
        self._lock = threading.Lock()
        # latest attitude in degrees: yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate
        self.attitude = None
        self.zoom = None
        self.on_attitude = None   # optional callback(attitude_tuple)
        self._running = True
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx.start()

    # ---------- low level ----------
    def send(self, cmd_id, data=b""):
        with self._lock:
            pkt = build_packet(cmd_id, data, self._seq)
            self._seq = (self._seq + 1) & 0xFFFF
            self.sock.sendto(pkt, self.addr)

    def _rx_loop(self):
        while self._running:
            try:
                buf, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            for cmd, data in parse_packets(buf):
                self._handle(cmd, data)

    def _handle(self, cmd, data):
        if cmd == CMD_GET_ATTITUDE and len(data) >= 12:
            vals = struct.unpack_from("<6h", data)
            self.attitude = tuple(v / 10.0 for v in vals)
            if self.on_attitude:
                self.on_attitude(self.attitude)
        elif cmd in (CMD_MANUAL_ZOOM, CMD_ABS_ZOOM) and len(data) >= 2:
            # reply: uint16 zoom x10
            (z,) = struct.unpack_from("<H", data)
            self.zoom = z / 10.0

    # ---------- high level ----------
    def set_angle(self, yaw_deg, pitch_deg):
        """A8 mini range: yaw -135..135, pitch -90..25 (negative = down)."""
        yaw_deg = max(-135.0, min(135.0, yaw_deg))
        pitch_deg = max(-90.0, min(25.0, pitch_deg))
        self.send(CMD_SET_ANGLE, struct.pack("<hh", int(yaw_deg * 10), int(pitch_deg * 10)))

    def set_speed(self, yaw_speed, pitch_speed):
        """-100..100 each; send 0,0 to stop."""
        clamp = lambda v: max(-100, min(100, int(v)))
        self.send(CMD_GIMBAL_SPEED, struct.pack("<bb", clamp(yaw_speed), clamp(pitch_speed)))

    def center(self):
        self.send(CMD_CENTER, b"\x01")

    def request_attitude(self):
        self.send(CMD_GET_ATTITUDE)

    def manual_zoom(self, direction):
        """1 = in, 0 = stop, -1 = out."""
        self.send(CMD_MANUAL_ZOOM, struct.pack("<b", int(direction)))

    def absolute_zoom(self, level):
        """A8 mini: 1.0 .. 6.0 (digital zoom)."""
        level = max(1.0, min(6.0, level))
        ip = int(level)
        dp = int(round((level - ip) * 10))
        self.send(CMD_ABS_ZOOM, struct.pack("<BB", ip, dp))

    def function(self, func):
        self.send(CMD_FUNCTION, struct.pack("<B", func))

    def close(self):
        self._running = False
        self.sock.close()
