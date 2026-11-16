#!/usr/bin/env python3
"""Score px4_control flight logs against the project goal.

Reads flight_logs/<YYYYMMDD_HHMMSS>/flight.csv (written by px4_control.flight_logger.FlightLogger),
computes metrics and a PASS/FAIL verdict with the criteria in tools/oa/goal_criteria.json.
Standard library only, no ROS needed, so it also runs on the Orin and from Codex.

Usage (from the workspace root):
    python3 tools/oa/flight_eval.py --latest                  # newest flight
    python3 tools/oa/flight_eval.py flight_logs/20261109_133500
    python3 tools/oa/flight_eval.py DIR --world worlds/my_forest.sdf   # SITL: closest obstacles
    python3 tools/oa/flight_eval.py DIR --json --write        # JSON on stdout, eval.json in DIR
    python3 tools/oa/flight_eval.py --summary DIR [DIR ...]   # one line per flight + PASS streak

Exit code: 0 all evaluated flights PASS, 1 at least one FAIL, 2 usage or read error.
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
CRITERIA_FILE = Path(__file__).resolve().parent / 'goal_criteria.json'
LOG_DIRS = [WS / 'flight_logs', Path.home() / 'ros2_px4_gz_ws' / 'flight_logs']
FOLDER_RE = re.compile(r'^\d{8}_\d{6}$')

NAV_STATES = {
    0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL', 3: 'AUTO_MISSION', 4: 'AUTO_LOITER', 5: 'AUTO_RTL',
    10: 'ACRO', 12: 'DESCEND', 13: 'TERMINATION', 14: 'OFFBOARD', 15: 'STAB', 17: 'AUTO_TAKEOFF',
    18: 'AUTO_LAND', 19: 'AUTO_FOLLOW_TARGET', 20: 'AUTO_PRECLAND', 21: 'ORBIT',
    22: 'AUTO_VTOL_TAKEOFF',
}
TEXT_COLUMNS = {'node', 'phase', 'height_src', 'vfh_mode'}
AIRBORNE_HEIGHT = 0.5      # m, rows above this while armed count as airborne
CLEARANCE_PHASES = {'hover', 'cruise'}
NOT_OBSTACLES = ('ground', 'sun', 'mark', 'grass', 'x500')


def load_criteria(path=CRITERIA_FILE):
    with open(path) as f:
        return json.load(f)


def find_flights(dirs=LOG_DIRS):
    """All flight folders, oldest first (folder names are timestamps)."""
    found = []
    for d in dirs:
        if d.is_dir():
            found += [p for p in d.iterdir()
                      if p.is_dir() and FOLDER_RE.match(p.name) and (p / 'flight.csv').exists()]
    return sorted(found, key=lambda p: p.name)


def read_rows(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if k is None:
                    continue
                if k in TEXT_COLUMNS:
                    row[k] = v
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            rows.append(row)
    return rows


def num(row, key):
    v = row.get(key)
    return v if isinstance(v, float) else math.nan


def ok(v):
    """Tell whether v is a measurement: not NaN (missing), not inf (nothing in range)."""
    return isinstance(v, float) and math.isfinite(v)


def world_obstacles(sdf_path):
    """Return (name, x_east, y_north) of every <include> of the world that is an obstacle."""
    world = ET.parse(sdf_path).getroot().find('world')
    out = []
    for inc in world.findall('include') if world is not None else []:
        uri = (inc.findtext('uri') or '').strip()
        name = (inc.findtext('name') or uri.split('/')[-1]).strip()
        if any(k in (name + uri).lower() for k in NOT_OBSTACLES):
            continue
        pose = (inc.findtext('pose') or '0 0 0 0 0 0').split()
        out.append((name, float(pose[0]), float(pose[1])))
    return out


def evaluate(folder, criteria=None, world=None):
    folder = Path(folder)
    csv_path = folder / 'flight.csv' if folder.is_dir() else folder
    folder = csv_path.parent
    crit = (criteria or load_criteria())['criteria']
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f'{csv_path} has no rows')
    cols = set(rows[0])

    node = rows[0].get('node') or ''
    if not node:
        try:    # automated trials (run_trial.sh) record the node
            node = json.loads((folder / 'trial.json').read_text())['node']
        except (OSError, ValueError, KeyError):
            pass
    if not node and (folder / 'vfh.npz').exists():
        # an obstacle_avoidance_static node: _gps flies Position mode and logs its sticks (until
        # 2026-11-16 the _gps name also flew the OFFBOARD code), _offboard flies OFFBOARD
        node = 'obstacle_avoidance_static_' + ('gps?' if 'stick_pitch' in cols else 'offboard?')
    node = node or 'unknown'

    t = [num(r, 't') for r in rows]
    # height above the ground when logged, else EKF altitude above the take-off point (up positive)
    height = [num(r, 'height') if ok(num(r, 'height')) else num(r, 'alt') for r in rows]
    armed = [num(r, 'armed') == 1.0 for r in rows]
    phase = [r.get('phase', '') for r in rows]
    airborne = [a and ok(h) and h > AIRBORNE_HEIGHT for a, h in zip(armed, height)]
    air_idx = [i for i, a in enumerate(airborne) if a]

    m = {'rows': len(rows),
         'duration_s': round(t[-1] - t[0], 1) if ok(t[0]) and ok(t[-1]) else None}
    m['airborne_s'] = round(t[air_idx[-1]] - t[air_idx[0]], 1) if air_idx else 0.0
    m['max_height_m'] = round(max((h for h in height if ok(h)), default=math.nan), 2)
    m['phases'] = list(dict.fromkeys(p for p in phase if p))
    m['final_phase'] = phase[-1]
    m['final_armed'] = armed[-1]
    m['landed'] = (not armed[-1]) and m['max_height_m'] >= 1.0

    # goal
    dg = [num(r, 'dist_goal') for r in rows]
    dg_ok = [d for d in dg if ok(d)]
    m['final_dist_goal_m'] = round(dg_ok[-1], 2) if dg_ok else None
    m['min_dist_goal_m'] = round(min(dg_ok), 2) if dg_ok else None

    # path
    path_len, prev = 0.0, None
    for i in air_idx:
        n, e = num(rows[i], 'north'), num(rows[i], 'east')
        if ok(n) and ok(e):
            if prev:
                path_len += math.hypot(n - prev[0], e - prev[1])
            prev = (n, e)
    m['path_length_m'] = round(path_len, 1)
    if air_idx and ok(num(rows[air_idx[0]], 'goal_n')):
        r0 = rows[air_idx[0]]
        straight = math.hypot(num(r0, 'goal_n') - num(r0, 'north'),
                              num(r0, 'goal_e') - num(r0, 'east'))
        m['straight_line_m'] = round(straight, 1)
        m['path_ratio'] = round(path_len / straight, 2) if straight > 1 else None

    speeds = [num(rows[i], 'speed') for i in air_idx if ok(num(rows[i], 'speed'))]
    m['max_speed_mps'] = round(max(speeds), 2) if speeds else None
    m['mean_speed_mps'] = round(sum(speeds) / len(speeds), 2) if speeds else None

    # clearance seen by the depth camera
    ccol = next((c for c in ('nearest_all', 'nearest_ahead') if c in cols), None)
    m['clearance_column'] = ccol
    if ccol:
        idx = [i for i in air_idx if phase[i] in CLEARANCE_PHASES] or air_idx
        vals = [(num(rows[i], ccol), i) for i in idx if ok(num(rows[i], ccol))]
        if vals:
            v, i = min(vals)
            m['min_clearance_m'] = round(v, 2)
            m['min_clearance_at_s'] = round(t[i], 1)
        else:
            m['min_clearance_m'] = None   # nothing within camera range during the flight
    holding = [num(r, 'holding') == 1.0 for r in rows]
    m['holds'] = sum(1 for a, b in zip(holding, holding[1:]) if b and not a)
    if 'vfh_mode' in cols:
        modes = {}
        for i in air_idx:
            mode = rows[i].get('vfh_mode') or '-'
            modes[mode] = modes.get(mode, 0) + 1
        m['vfh_modes'] = modes

    # PX4 modes
    seq = []
    for r, a in zip(rows, armed):
        s = num(r, 'nav_state')
        if ok(s) and (not seq or seq[-1] != int(s)):
            seq.append(int(s))
    m['nav_states'] = [f'{s}:{NAV_STATES.get(s, "?")}' for s in seq]
    armed_states = {int(num(r, 'nav_state')) for r, a in zip(rows, armed)
                    if a and ok(num(r, 'nav_state'))}
    bad_states = sorted(armed_states & set(crit['forbidden_nav_states']))

    # crash heuristics
    crashes = []
    for i in range(1, len(rows)):
        h = height[i - 1]
        if armed[i - 1] and not armed[i] and ok(h) and h > crit['disarm_in_air_height_m']:
            crashes.append(f'disarmed at {h:.1f} m (t={t[i]:.1f} s)')
            break
    for i in air_idx:
        vd = num(rows[i], 'vd')
        if ok(vd) and vd > crit['crash_descent_mps'] and phase[i] not in ('landing', 'stopping'):
            crashes.append(f'fell at {vd:.1f} m/s in phase {phase[i]} (t={t[i]:.1f} s)')
            break
    if m['max_speed_mps'] is not None and m['max_speed_mps'] > crit['crash_speed_mps']:
        crashes.append(f'speed {m["max_speed_mps"]} m/s > {crit["crash_speed_mps"]} m/s')
    m['crash_signs'] = crashes

    # SITL ground truth: closest obstacles of the world (path in ENU: x = east, y = north)
    if world:
        obs = world_obstacles(world)
        best = {}
        for i in air_idx:
            n, e = num(rows[i], 'north'), num(rows[i], 'east')
            if not (ok(n) and ok(e)):
                continue
            for name, ox, oy in obs:
                d = math.hypot(e - ox, n - oy)
                if d < best.get(name, (math.inf,))[0]:
                    best[name] = (d, round(t[i], 1))
        m['world'] = str(world)
        m['closest_obstacles'] = [{'name': k, 'center_dist_m': round(v[0], 2), 't_s': v[1]}
                                  for k, v in sorted(best.items(), key=lambda kv: kv[1][0])[:3]]

    # verdict
    fails, notes = [], []
    if m['final_dist_goal_m'] is None:
        notes.append('no dist_goal column: goal criterion not applicable')
    elif m['final_dist_goal_m'] > crit['goal_radius_m']:
        fails.append(f'ended {m["final_dist_goal_m"]} m from the goal '
                     f'(> {crit["goal_radius_m"]} m)')
    if crit.get('must_land') and not m['landed']:
        fails.append('did not land and disarm' if m['max_height_m'] >= 1.0 else 'never took off')
    if ccol is None:
        notes.append('no nearest_* column: clearance not checked')
    elif m.get('min_clearance_m') is not None and m['min_clearance_m'] < crit['min_clearance_m']:
        fails.append(f'clearance {m["min_clearance_m"]} m at t={m["min_clearance_at_s"]} s '
                     f'(< {crit["min_clearance_m"]} m)')
    if m['airborne_s'] > crit['max_airborne_s']:
        fails.append(f'airborne {m["airborne_s"]} s (> {crit["max_airborne_s"]} s)')
    if bad_states:
        fails.append('failsafe nav_state '
                     + ', '.join(f'{s}:{NAV_STATES.get(s)}' for s in bad_states))
    fails += [f'crash sign: {c}' for c in crashes]

    return {
        'folder': str(folder), 'node': node, 'verdict': 'FAIL' if fails else 'PASS',
        'failures': fails, 'notes': notes, 'metrics': m, 'criteria': crit,
        'evaluated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'evaluator': 'tools/oa/flight_eval.py',
    }


def format_report(res):
    m = res['metrics']
    lines = [f'{res["verdict"]}  {res["folder"]}  ({res["node"]})']
    lines += [f'  FAIL: {f}' for f in res['failures']]
    lines += [f'  note: {n}' for n in res['notes']]
    lines.append(f'  airborne {m["airborne_s"]} s, max height {m["max_height_m"]} m, '
                 f'landed {m["landed"]}, final phase {m["final_phase"]}')
    lines.append(f'  goal: final {m["final_dist_goal_m"]} m, min {m["min_dist_goal_m"]} m; path '
                 f'{m["path_length_m"]} m (ratio {m.get("path_ratio")})')
    lines.append(f'  clearance ({m["clearance_column"]}): min {m.get("min_clearance_m")} m '
                 f'at t={m.get("min_clearance_at_s")} s; holds {m["holds"]}; '
                 f'speed max {m["max_speed_mps"]} / mean {m["mean_speed_mps"]} m/s')
    lines.append(f'  nav_state: {" > ".join(m["nav_states"])}')
    if m.get('vfh_modes'):
        lines.append(f'  vfh_mode rows: {m["vfh_modes"]}')
    for o in m.get('closest_obstacles', []):
        lines.append(f'  closest obstacle {o["name"]}: {o["center_dist_m"]} m from its center '
                     f'at t={o["t_s"]} s')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('folders', nargs='*', help='flight folders (or flight.csv files)')
    ap.add_argument('--latest', action='store_true', help='evaluate the newest flight folder')
    ap.add_argument('--summary', action='store_true',
                    help='one line per flight plus the PASS streak')
    ap.add_argument('--world',
                    help='Gazebo .sdf the flight used (SITL): report the closest obstacles')
    ap.add_argument('--criteria', default=str(CRITERIA_FILE))
    ap.add_argument('--json', action='store_true', help='print JSON')
    ap.add_argument('--write', action='store_true',
                    help='also write eval.json into each flight folder')
    args = ap.parse_args(argv)

    folders = [Path(f) for f in args.folders]
    if args.latest:
        flights = find_flights()
        if not flights:
            print('no flight folders found in ' + ', '.join(map(str, LOG_DIRS)), file=sys.stderr)
            return 2
        folders.append(flights[-1])
    if not folders:
        ap.print_usage(sys.stderr)
        return 2

    criteria = load_criteria(args.criteria)
    results = []
    for f in folders:
        try:
            res = evaluate(f, criteria, args.world)
        except (OSError, ValueError) as exc:
            print(f'{f}: {exc}', file=sys.stderr)
            if len(folders) == 1:
                return 2
            continue   # e.g. a simple mission that wrote <node>.csv instead of flight.csv
        results.append(res)
        if args.write:
            with open(Path(res['folder']) / 'eval.json', 'w') as out:
                json.dump(res, out, indent=2)

    if not results:
        return 2
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    elif args.summary:
        streak = 0
        for res in results:
            m = res['metrics']
            streak = streak + 1 if res['verdict'] == 'PASS' else 0
            print(f'{res["verdict"]:4}  {os.path.basename(res["folder"])}  '
                  f'goal {m["final_dist_goal_m"]} m  '
                  f'clear {m.get("min_clearance_m")} m  air {m["airborne_s"]} s  '
                  f'{"; ".join(res["failures"])[:90]}')
        need = criteria['acceptance']['consecutive_passes']
        print(f'PASS streak at the end: {streak} (goal: {need})')
    else:
        print('\n\n'.join(format_report(r) for r in results))
    return 0 if all(r['verdict'] == 'PASS' for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
