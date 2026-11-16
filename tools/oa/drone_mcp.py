#!/usr/bin/env python3
"""drone: read-only MCP server (stdio) for the PX4 depth-camera obstacle-avoidance workspace.

Tools: stack_status, ros_graph, vehicle_state, depth_snapshot, topic_rates, flight_logs,
flight_report. It never commands the vehicle: no publisher, no service call, no MAVLink.
Its only writes are eval.json in a flight folder (flight_report) and PNG snapshots in
~/.cache/drone-oa/snapshots.

Start it through tools/oa/drone_mcp.sh, which sources ROS 2 and the workspace install.
Claude Code: plugin .claude/skills/drone-oa (.mcp.json). Codex: `codex mcp add drone -- bash
<workspace>/tools/oa/drone_mcp.sh`. Protocol: JSON-RPC 2.0, one message per line.
"""

import base64
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.path.insert(0, str(HERE))
import flight_eval  # noqa: E402

OUT = sys.stdout
sys.stdout = sys.stderr          # library prints must never reach the protocol stream
WRITE_LOCK = threading.Lock()
SUPPORTED = ('2025-06-18', '2025-03-26', '2024-11-05')
CACHE = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'drone-oa'
SNAP_DIR = CACHE / 'snapshots'
SITL_PIDS = CACHE / 'sitl' / 'pids'
DEFAULT_DOMAIN = int(os.environ.get('ROS_DOMAIN_ID', '0') or 0)
ARMING = {0: 'INIT', 1: 'STANDBY', 2: 'ARMED', 3: 'STANDBY_ERROR', 4: 'SHUTDOWN'}

INSTRUCTIONS = (
    'Read-only view of the PX4 SITL/real-drone stack and its flight logs. domain_id selects '
    'the ROS 2 domain: 42 for the isolated headless stack of tools/oa/sitl.sh, otherwise the '
    'default '
    f'({DEFAULT_DOMAIN}) for the usual GUI stack. Call stack_status first to see what runs. '
    'Live tools need the stack running and wait at most timeout_s for messages.')

DOMAIN_PROP = {'type': 'integer', 'description': f'ROS 2 domain (default {DEFAULT_DOMAIN}; '
               '42 = tools/oa/sitl.sh headless stack)'}
NS_PROP = {'type': 'string', 'description': "PX4 topic namespace, '' for /fmu/..., e.g. 'px4_1'"}
TIMEOUT_PROP = {'type': 'number', 'description': 'seconds to wait for messages (default 3)'}

TOOLS = [
    {'name': 'stack_status',
     'description': 'Which simulator / PX4 / agent / bridge / px4_control processes run, with '
                    'their ROS_DOMAIN_ID and GZ_PARTITION, whether they belong to '
                    'tools/oa/sitl.sh, and '
                    'the newest flight log with its verdict. No ROS needed.',
     'inputSchema': {'type': 'object', 'properties': {}}},
    {'name': 'ros_graph',
     'description': 'ROS 2 topics (with types) and nodes visible in a domain.',
     'inputSchema': {'type': 'object', 'properties': {'domain_id': DOMAIN_PROP,
                                                      'filter': {'type': 'string',
                                                                 'description': 'substring'}}}},
    {'name': 'vehicle_state',
     'description': 'One reading of PX4 vehicle_status, vehicle_local_position (NED, '
                    'dist_bottom) and vehicle_global_position, plus distance and bearing to the '
                    'goal (.json in the '
                    'format of worlds/*_goal.json; default ~/.config/px4_control/gps_point.json).',
     'inputSchema': {'type': 'object', 'properties': {
         'domain_id': DOMAIN_PROP, 'namespace': NS_PROP, 'timeout_s': TIMEOUT_PROP,
         'goal': {'type': 'string', 'description': 'goal .json path'}}}},
    {'name': 'depth_snapshot',
     'description': 'Grab one depth image: valid-pixel share, nearest point with its direction, '
                    'minimum depth in a 3x3 grid, and a colour PNG (near = red, no return = '
                    'black) returned inline and saved under ~/.cache/drone-oa/snapshots.',
     'inputSchema': {'type': 'object', 'properties': {
         'domain_id': DOMAIN_PROP, 'timeout_s': TIMEOUT_PROP,
         'topic': {'type': 'string', 'description': 'default /camera/depth/image_raw'},
         'depth_min': {'type': 'number', 'description': 'ignore closer returns (m), default 0.3'},
         'max_range': {'type': 'number',
                       'description': 'ignore farther returns (m), default 12'}}}},
    {'name': 'topic_rates',
     'description': 'Measure message rates (Hz) of topics for a few seconds: depth image, camera '
                    'info, PX4 status/position, and the setpoints a running node publishes.',
     'inputSchema': {'type': 'object', 'properties': {
         'domain_id': DOMAIN_PROP, 'seconds': {'type': 'number', 'description': 'default 3'},
         'topics': {'type': 'array', 'items': {'type': 'string'}}}}},
    {'name': 'flight_logs',
     'description': 'Newest flight folders of flight_logs/ (this and the gz workspace) with node, '
                    'airborne time and verdict when already scored (eval.json).',
     'inputSchema': {'type': 'object', 'properties': {
         'limit': {'type': 'integer', 'description': 'default 10'}}}},
    {'name': 'flight_report',
     'description': 'Score one flight against tools/oa/goal_criteria.json (PASS/FAIL, goal '
                    'distance, clearance, nav_state, crash signs; closest world obstacles when '
                    'world is '
                    'given) and write eval.json into its folder.',
     'inputSchema': {'type': 'object', 'properties': {
         'folder': {'type': 'string', 'description': 'flight folder; default the newest'},
         'world': {'type': 'string', 'description': 'e.g. worlds/my_forest.sdf (SITL only)'},
         'write': {'type': 'boolean', 'description': 'write eval.json (default true)'}}}},
]


# ---------------------------------------------------------------- processes and logs (no ROS)

PATTERNS = (('gz sim', 'gazebo'), ('bin/px4', 'px4'), ('MicroXRCEAgent', 'uxrce-agent'),
            ('parameter_bridge', 'gz-bridge'), ('px4_control', 'px4_control node'),
            ('depth_camera', 'depth_camera node'), ('mavproxy', 'mavproxy'),
            ('mavlink-routerd', 'mavlink-router'))


def proc_env(pid, keys=('ROS_DOMAIN_ID', 'GZ_PARTITION')):
    try:
        raw = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except OSError:
        return {}
    env = dict(e.decode(errors='replace').split('=', 1) for e in raw if b'=' in e)
    return {k: env[k] for k in keys if k in env}


def tool_stack_status(_args):
    ours = set()
    if SITL_PIDS.exists():
        for line in SITL_PIDS.read_text().split('\n'):
            if line.strip():
                pgid = line.split()[1]
                out = subprocess.run(['pgrep', '-g', pgid], capture_output=True, text=True)
                ours.update(out.stdout.split())
    ps = subprocess.run(['ps', '-eo', 'pid=,etimes=,args='], capture_output=True, text=True).stdout
    lines = []
    for row in ps.splitlines():
        pid, etimes, args = (row.strip().split(None, 2) + ['', ''])[:3]
        label = next((lab for pat, lab in PATTERNS if pat in args), None)
        shell = args.startswith(('bash -c', '/bin/bash'))
        if not label or shell or 'drone_mcp' in args or 'pgrep' in args:
            continue
        env = proc_env(pid)
        owner = 'tools/oa/sitl.sh' if pid in ours else 'user/other'
        lines.append(f'{label:17} pid {pid:>7}  up {int(etimes) // 60} min  '
                     f'domain {env.get("ROS_DOMAIN_ID", "-(0)")}  '
                     f'partition {env.get("GZ_PARTITION", "-")}  [{owner}]  {args[:110]}')
    text = ['Processes:'] + (lines or ['  none (no simulator, PX4, agent or node running)'])
    flights = flight_eval.find_flights()
    if flights:
        text.append('Newest flight: ' + describe_flight(flights[-1]))
    return [txt('\n'.join(text))]


def describe_flight(folder):
    ev = folder / 'eval.json'
    if ev.exists():
        try:
            res = json.loads(ev.read_text())
            m = res['metrics']
            return (f'{folder} {res["node"]}: {res["verdict"]} (goal {m["final_dist_goal_m"]} m, '
                    f'clearance {m.get("min_clearance_m")} m, airborne {m["airborne_s"]} s)'
                    + (f' - {"; ".join(res["failures"])}' if res['failures'] else ''))
        except (ValueError, KeyError):
            pass
    try:
        with open(folder / 'flight.csv') as f:
            header = f.readline()
            first = f.readline().split(',')
        node = first[0] if header.startswith('node,') else '?'
    except OSError:
        node = '?'
    return f'{folder} {node}: not scored yet'


def tool_flight_logs(args):
    flights = flight_eval.find_flights()[-int(args.get('limit') or 10):]
    if not flights:
        dirs = ', '.join(map(str, flight_eval.LOG_DIRS))
        return [txt(f'No flight folders with flight.csv in {dirs}')]
    return [txt('\n'.join(describe_flight(f) for f in reversed(flights)))]


def tool_flight_report(args):
    folder = args.get('folder')
    if folder:
        folder = Path(folder) if Path(folder).is_absolute() else WS / folder
    else:
        flights = flight_eval.find_flights()
        if not flights:
            raise RuntimeError('no flight folders found')
        folder = flights[-1]
    world = args.get('world')
    if world and not Path(world).is_absolute():
        world = str(WS / world)
    res = flight_eval.evaluate(folder, flight_eval.load_criteria(), world)
    if args.get('write', True):
        (Path(res['folder']) / 'eval.json').write_text(json.dumps(res, indent=2))
    plots = [p.name for p in Path(res['folder']).glob('*.png')]
    extra = ('\nPlots in the folder (Read them to look): ' + ', '.join(sorted(plots))
             if plots else '')
    return [txt(flight_eval.format_report(res) + extra)]


# ---------------------------------------------------------------- ROS 2 (rclpy, lazy)

class Ros:
    def __init__(self):
        self.lock = threading.Lock()
        self.domains = {}

    def node(self, domain):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        if domain not in self.domains:
            ctx = Context()
            rclpy.init(context=ctx, domain_id=domain,
                       signal_handler_options=SignalHandlerOptions.NO)
            node = rclpy.create_node(f'drone_mcp_{os.getpid()}', context=ctx)
            ex = SingleThreadedExecutor(context=ctx)
            ex.add_node(node)
            threading.Thread(target=ex.spin, daemon=True).start()
            self.domains[domain] = node
            time.sleep(1.5)       # DDS discovery
        return self.domains[domain]

    @staticmethod
    def qos():
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.VOLATILE,
                          history=HistoryPolicy.KEEP_LAST, depth=5)

    def latest(self, domain, wanted, timeout):
        """Wait for one message per topic: {key: (topic, type)} -> {key: (msg, t) or None}."""
        node = self.node(domain)
        got, subs = {}, []
        for key, (topic, mtype) in wanted.items():
            def cb(msg, key=key):
                got[key] = (msg, time.time())
            subs.append(node.create_subscription(mtype, topic, cb, self.qos()))
        end = time.time() + timeout
        while time.time() < end and len(got) < len(wanted):
            time.sleep(0.05)
        for s in subs:
            node.destroy_subscription(s)
        return {k: got.get(k) for k in wanted}


ROS = Ros()


def domain_of(args):
    d = args.get('domain_id')
    return DEFAULT_DOMAIN if d is None else int(d)


def tool_ros_graph(args):
    with ROS.lock:
        node = ROS.node(domain_of(args))
        topics = node.get_topic_names_and_types()
        nodes = node.get_node_names_and_namespaces()
    flt = args.get('filter') or ''
    lines = [f'domain {domain_of(args)}: {len(topics)} topics, {len(nodes)} nodes']
    lines += [f'  {name}  [{", ".join(types)}]' for name, types in sorted(topics) if flt in name]
    lines.append('nodes: ' + ', '.join(sorted(f'{ns.rstrip("/")}/{n}' for n, ns in nodes)))
    return [txt('\n'.join(lines))]


def load_goal(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = WS / path
    data = json.loads(path.read_text())
    entry = data[0] if isinstance(data, list) else data
    gp = entry.get('global_position', entry)
    return float(gp['lat']), float(gp['lon']), str(path)


def tool_vehicle_state(args):
    from px4_msgs.msg import VehicleGlobalPosition, VehicleLocalPosition, VehicleStatus
    ns = (args.get('namespace') or '').strip('/')
    pre = f'/{ns}/fmu/out/' if ns else '/fmu/out/'
    wanted = {'status': (pre + 'vehicle_status', VehicleStatus),
              'local': (pre + 'vehicle_local_position', VehicleLocalPosition),
              'global': (pre + 'vehicle_global_position', VehicleGlobalPosition)}
    with ROS.lock:
        got = ROS.latest(domain_of(args), wanted, float(args.get('timeout_s') or 3.0))
    lines = [f'domain {domain_of(args)}, topics {pre}*']
    now = time.time()
    st = got['status']
    if st:
        s = st[0]
        nav = flight_eval.NAV_STATES.get(s.nav_state, '?')
        lines.append(f'status: nav_state {s.nav_state} {nav}, arming {s.arming_state} '
                     f'{ARMING.get(s.arming_state, "?")}, failsafe {s.failsafe}, '
                     f'preflight checks pass {s.pre_flight_checks_pass}')
    else:
        lines.append(f'status: no {wanted["status"][0]} message '
                     '(is PX4 + MicroXRCEAgent up in this domain?)')
    lp = got['local']
    if lp:
        p = lp[0]
        lines.append(f'local NED: x(N) {p.x:.2f}  y(E) {p.y:.2f}  z(D) {p.z:.2f} m  v {p.vx:.2f} '
                     f'{p.vy:.2f} {p.vz:.2f} m/s  heading {math.degrees(p.heading):.1f} deg  '
                     f'xy_valid {p.xy_valid} z_valid {p.z_valid}')
        lines.append(f'dist_bottom {p.dist_bottom:.2f} m (valid {p.dist_bottom_valid})  '
                     f'age {now - lp[1]:.1f} s')
    else:
        lines.append('local position: no message')
    gp = got['global']
    if gp:
        g = gp[0]
        lines.append(f'global: lat {g.lat:.7f} lon {g.lon:.7f} alt {g.alt:.1f} m AMSL '
                     f'(eph {g.eph:.1f} m)')
        goal = args.get('goal') or str(Path.home() / '.config/px4_control/gps_point.json')
        try:
            glat, glon, gpath = load_goal(goal)
            dn = math.radians(glat - g.lat) * 6371000.0
            de = math.radians(glon - g.lon) * 6371000.0 * math.cos(math.radians(g.lat))
            lines.append(f'goal {gpath}: {math.hypot(dn, de):.1f} m away, bearing '
                         f'{math.degrees(math.atan2(de, dn)) % 360:.0f} deg')
        except (OSError, ValueError, KeyError, IndexError) as exc:
            lines.append(f'goal: cannot read {goal} ({exc})')
    else:
        lines.append('global position: no message')
    return [txt('\n'.join(lines))]


def depth_array(msg):
    import numpy as np
    enc = msg.encoding.lower()
    if enc in ('32fc1', 'tc1'):
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        scale = 1.0
    elif enc in ('16uc1', 'mono16'):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        scale = 0.001
    else:
        raise RuntimeError(f'unsupported depth encoding {msg.encoding}')
    if msg.is_bigendian:
        arr = arr.byteswap()
    per_row = msg.step // arr.itemsize
    return arr.reshape(msg.height, per_row)[:, :msg.width].astype(np.float32) * scale


def colorize(depth, valid, depth_min, max_range):
    import numpy as np
    t = np.clip((max_range - np.nan_to_num(depth, nan=max_range)) / (max_range - depth_min), 0, 1)
    stops = [0.0, 0.35, 0.7, 1.0]   # far blue -> green -> yellow -> near red
    rgb = np.stack([np.interp(t, stops, c) for c in ((30, 40, 230, 255), (40, 200, 230, 30),
                                                     (120, 80, 30, 30))], axis=-1)
    rgb[~valid] = 0
    return rgb.astype(np.uint8)


def tool_depth_snapshot(args):
    import numpy as np
    from sensor_msgs.msg import CameraInfo, Image
    topic = args.get('topic') or '/camera/depth/image_raw'
    info_topic = topic.rsplit('/', 1)[0] + '/camera_info'
    dmin = float(args.get('depth_min') or 0.3)
    dmax = float(args.get('max_range') or 12.0)
    with ROS.lock:
        wanted = {'img': (topic, Image), 'info': (info_topic, CameraInfo)}
        got = ROS.latest(domain_of(args), wanted, float(args.get('timeout_s') or 3.0))
    if not got['img']:
        raise RuntimeError(f'no image on {topic} in domain {domain_of(args)} '
                           '(bridge or camera running?)')
    msg, t_rx = got['img']
    depth = depth_array(msg)
    valid = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    h, w = depth.shape
    lines = [f'{topic}: {w}x{h} {msg.encoding}, frame {msg.header.frame_id}, received '
             f'{time.time() - t_rx:.1f} s ago; valid {100 * valid.mean():.1f}% '
             f'({dmin} m < depth < {dmax} m)']
    if valid.any():
        vals = np.where(valid, depth, np.inf)
        v, u = np.unravel_index(np.argmin(vals), vals.shape)
        d = float(vals[v, u])
        where = f'pixel ({u}, {v})'
        if got['info']:
            k = got['info'][0].k
            fx, fy, cx, cy = k[0], k[4], k[2], k[5]
            if fx and fy:
                az = math.degrees(math.atan2(u - cx, fx))
                el = -math.degrees(math.atan2(v - cy, fy))
                where += f', {az:+.0f} deg right, {el:+.0f} deg up of the optical axis'
        median = float(np.median(depth[valid]))
        lines.append(f'nearest {d:.2f} m at {where}; median {median:.2f} m')
    else:
        lines.append('no valid return in range')
    grid = []
    for r, rname in enumerate(('top', 'middle', 'bottom')):
        cells = []
        for c, cname in enumerate(('left', 'center', 'right')):
            sl = (slice(r * h // 3, (r + 1) * h // 3), slice(c * w // 3, (c + 1) * w // 3))
            cell = depth[sl][valid[sl]]
            cells.append(f'{cname} {cell.min():.2f}' if cell.size else f'{cname}  -  ')
        grid.append(f'  {rname:6}: ' + ' | '.join(cells))
    lines.append('min depth (m) per region:\n' + '\n'.join(grid))

    content = []
    try:
        from PIL import Image as Pil, ImageDraw
        img = Pil.fromarray(colorize(depth, valid, dmin, dmax), 'RGB')
        draw = ImageDraw.Draw(img)
        for i in (1, 2):
            draw.line([(i * w // 3, 0), (i * w // 3, h)], fill=(255, 255, 255))
            draw.line([(0, i * h // 3), (w, i * h // 3)], fill=(255, 255, 255))
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAP_DIR / time.strftime('%Y%m%d_%H%M%S_depth.png')
        img.save(path)
        small = img.resize((320, max(1, int(320 * h / w)))) if w > 320 else img
        buf = io.BytesIO()
        small.save(buf, format='PNG')
        lines.append(f'saved {path}')
        content.append({'type': 'image', 'data': base64.b64encode(buf.getvalue()).decode(),
                        'mimeType': 'image/png'})
    except ImportError:
        lines.append('(PIL missing: no picture)')
    return [txt('\n'.join(lines))] + content


DEFAULT_RATE_TOPICS = ['/camera/depth/image_raw', '/camera/depth/camera_info',
                       '/fmu/out/vehicle_status', '/fmu/out/vehicle_local_position',
                       '/fmu/out/vehicle_global_position', '/fmu/in/offboard_control_mode',
                       '/fmu/in/trajectory_setpoint', '/fmu/in/vehicle_command']


def tool_topic_rates(args):
    from rosidl_runtime_py.utilities import get_message
    topics = args.get('topics') or DEFAULT_RATE_TOPICS
    secs = float(args.get('seconds') or 3.0)
    with ROS.lock:
        node = ROS.node(domain_of(args))
        types = dict(node.get_topic_names_and_types())
        counts, subs, missing = {}, [], []
        for topic in topics:
            if topic not in types:
                missing.append(topic)
                continue
            counts[topic] = 0

            def cb(_msg, topic=topic):
                counts[topic] += 1
            mtype = get_message(types[topic][0])
            subs.append(node.create_subscription(mtype, topic, cb, ROS.qos()))
        time.sleep(secs)
        for s in subs:
            node.destroy_subscription(s)
    lines = [f'domain {domain_of(args)}, {secs:.0f} s:']
    lines += [f'  {t}: {c / secs:.1f} Hz' for t, c in counts.items()]
    lines += [f'  {t}: not in the graph' for t in missing]
    return [txt('\n'.join(lines))]


# ---------------------------------------------------------------- protocol

HANDLERS = {'stack_status': tool_stack_status, 'ros_graph': tool_ros_graph,
            'vehicle_state': tool_vehicle_state, 'depth_snapshot': tool_depth_snapshot,
            'topic_rates': tool_topic_rates, 'flight_logs': tool_flight_logs,
            'flight_report': tool_flight_report}


def txt(s):
    return {'type': 'text', 'text': s}


def send(msg):
    with WRITE_LOCK:
        OUT.write(json.dumps(msg) + '\n')
        OUT.flush()


def call_tool(req_id, params):
    name = params.get('name')
    args = params.get('arguments') or {}
    try:
        if name not in HANDLERS:
            raise RuntimeError(f'unknown tool {name}')
        result = {'content': HANDLERS[name](args), 'isError': False}
    except Exception as exc:   # report every failure to the client instead of dying
        traceback.print_exc()
        result = {'content': [txt(f'{name} failed: {exc}')], 'isError': True}
    send({'jsonrpc': '2.0', 'id': req_id, 'result': result})


def handle(req):
    method, req_id = req.get('method'), req.get('id')
    if method == 'initialize':
        asked = (req.get('params') or {}).get('protocolVersion')
        send({'jsonrpc': '2.0', 'id': req_id, 'result': {
            'protocolVersion': asked if asked in SUPPORTED else SUPPORTED[0],
            'capabilities': {'tools': {'listChanged': False}},
            'serverInfo': {'name': 'drone', 'version': '0.1.0'},
            'instructions': INSTRUCTIONS}})
    elif method == 'tools/list':
        send({'jsonrpc': '2.0', 'id': req_id, 'result': {'tools': TOOLS}})
    elif method == 'tools/call':
        threading.Thread(target=call_tool, args=(req_id, req.get('params') or {}),
                         daemon=True).start()
    elif method == 'ping':
        send({'jsonrpc': '2.0', 'id': req_id, 'result': {}})
    elif req_id is not None and not str(method).startswith('notifications/'):
        send({'jsonrpc': '2.0', 'id': req_id,
              'error': {'code': -32601, 'message': f'method not found: {method}'}})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            send({'jsonrpc': '2.0', 'id': None,
                  'error': {'code': -32700, 'message': 'parse error'}})
            continue
        for one in req if isinstance(req, list) else [req]:
            handle(one)


if __name__ == '__main__':
    main()
