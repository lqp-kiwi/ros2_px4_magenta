#!/usr/bin/env bash
# Obstacle-avoidance test suite on the headless SITL stack: three worlds, each without and with
# wind, in one call. Every case is one tools/oa/run_trial.sh on a fresh stack.
#
#   tools/oa/wind_suite.sh [--wind E,N] [--gust G] [--cases tree,grass,forest] [--repeat N]
#                          [--calm-only | --wind-only] [--node NAME] [--timeout S]
#
#   tree    my_world     worlds/tree_goal.json    one oak on the straight line, goal 44 m east
#   grass   grass_field  worlds/grass_goal.json   grass about 0.5 m tall, goal 20 m east
#   forest  my_forest    worlds/forest_goal.json  forest, goal 52 m east
#
# Wind (default --wind 0,5 --gust 0.3): the air moves at 5 m/s towards north, across the routes
# (they all go east), its speed varying +-30 % over 8 s; --gust 0 keeps it constant. Gazebo
# applies it as rotor drag only (tools/oa/wind_world.py), the tilt columns show how much it pushed.
# The table is also written to ~/.cache/drone-oa/suites/<date>_<time>/summary.txt, next to the
# output of every trial. The 6 default trials take about 17 min.
# Ctrl+C: the running node switches PX4 to Hold and saves its log, then the suite stops and prints
# what it has. Exit code: 0 every trial PASS, 1 a FAIL, 2 a setup error, 130 interrupted.
# Runs only on the headless stack: never next to the GUI stack, never on a vehicle.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WIND=0,5
GUST=0.3
CASES=tree,grass,forest
CONDITIONS="calm wind"
REPEAT=1
NODE=obstacle_avoidance_static_offboard
TIMEOUT=900

die() { echo "wind_suite.sh: $*" >&2; exit 2; }
need() { [ $# -ge 2 ] || die "$1 needs a value"; }
while [ $# -gt 0 ]; do
    case $1 in
        --wind) need "$@"; WIND=$2; shift 2 ;;
        --gust) need "$@"; GUST=$2; shift 2 ;;
        --cases) need "$@"; CASES=$2; shift 2 ;;
        --repeat) need "$@"; REPEAT=$2; shift 2 ;;
        --calm-only) CONDITIONS=calm; shift ;;
        --wind-only) CONDITIONS=wind; shift ;;
        --node) need "$@"; NODE=$2; shift 2 ;;
        --timeout) need "$@"; TIMEOUT=$2; shift 2 ;;
        -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done

declare -A WORLD_OF=([tree]=my_world [grass]=grass_field [forest]=my_forest)
declare -A GOAL_OF=([tree]=worlds/tree_goal.json [grass]=worlds/grass_goal.json
                    [forest]=worlds/forest_goal.json)
IFS=, read -ra case_list <<< "$CASES"
read -ra cond_list <<< "$CONDITIONS"
for c in "${case_list[@]}"; do
    [ -n "${WORLD_OF[$c]:-}" ] || die "unknown case '$c' (tree, grass, forest)"
done
[[ $REPEAT =~ ^[1-9][0-9]*$ ]] || die "--repeat wants a positive number"

cd "$WS" || exit 2
# The suite owns the headless stack for the whole run: refuse next to any running simulator
stack_status=$(tools/oa/sitl.sh status)
if grep -qE '^  [a-z]+  pid [0-9]+  running$|^other simulator' <<< "$stack_status"; then
    echo "$stack_status" >&2
    die "a simulator is running: close the GUI stack, or stop the headless one (tools/oa/sitl.sh down)"
fi

stamp=$(date +%Y%m%d_%H%M%S)
SUITE="${XDG_CACHE_HOME:-$HOME/.cache}/drone-oa/suites/$stamp"
mkdir -p "$SUITE"
total=$(( ${#case_list[@]} * ${#cond_list[@]} * REPEAT ))
echo "wind_suite.sh: $total trials of $NODE (wind $WIND m/s, gust $GUST), output in $SUITE"

stop=0
trap 'stop=1' INT TERM
results=()
n=0
for c in "${case_list[@]}"; do
    for cond in "${cond_list[@]}"; do
        for ((i = 1; i <= REPEAT; i++)); do
            [ "$stop" = 0 ] || break 3
            n=$((n + 1))
            args=(--node "$NODE" --world "${WORLD_OF[$c]}" --goal "${GOAL_OF[$c]}" --timeout "$TIMEOUT")
            what="no wind"
            if [ "$cond" = wind ]; then
                args+=(--wind "$WIND" --gust "$GUST")
                what="wind $WIND m/s, gust $GUST"
            fi
            log="$SUITE/$(printf '%02d' "$n")_${c}_${cond}.log"
            printf '\n=== [%d/%d] %s (%s), %s\n' "$n" "$total" "$c" "${WORLD_OF[$c]}" "$what"
            # tee -i: Ctrl+C must not kill the pipe while run_trial.sh lands the drone and scores
            tools/oa/run_trial.sh "${args[@]}" 2>&1 | tee -i "$log"
            rc=${PIPESTATUS[0]}
            folder=$(sed -n 's/^run_trial.sh: flight folder //p' "$log" | tail -1)
            results+=("$c|$cond|$rc|${folder:--}")
            tools/oa/sitl.sh down > /dev/null 2>&1 || true    # next case on a fresh stack
        done
    done
done
trap - INT TERM

python3 - "$SUITE/summary.txt" "$WIND" "$GUST" "${results[@]}" <<'EOF'
import json
import os
import sys

import numpy as np

out, wind, gust = sys.argv[1:4]
items = [s.split('|') for s in sys.argv[4:]]


def tilt_deg(ulg):
    """Median and 95th percentile of the tilt in the air (deg), from the PX4 ULog; None without
    pyulog. Holding position in wind tilts the drone into it."""
    try:
        from pyulog import ULog
        log = ULog(ulg, ['vehicle_attitude', 'vehicle_land_detected'])
        data = {d.name: d.data for d in log.data_list}
        att, land = data['vehicle_attitude'], data['vehicle_land_detected']
    except Exception:
        return None
    i = np.searchsorted(land['timestamp'], att['timestamp'], side='right') - 1
    air = (i >= 0) & (land['landed'][np.clip(i, 0, None)] == 0)
    x, y = att['q[1]'][air], att['q[2]'][air]
    if not len(x):
        return None
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    return float(np.median(tilt)), float(np.percentile(tilt, 95))


def fmt(v, digits=1):
    return '-' if v is None else f'{v:.{digits}f}' if isinstance(v, float) else str(v)


header = ['case', 'wind', 'verdict', 'goal m', 'clear m', 'air s', 'path', 'holds',
          'tilt med', 'tilt p95', 'flight folder']
rows, notes = [], []
for case, cond, rc, folder in items:
    ev, tilt = {}, None
    if folder != '-':
        if os.path.exists(os.path.join(folder, 'eval.json')):
            with open(os.path.join(folder, 'eval.json')) as f:
                ev = json.load(f)
        if os.path.exists(os.path.join(folder, 'px4.ulg')):
            tilt = tilt_deg(os.path.join(folder, 'px4.ulg'))
    m = ev.get('metrics', {})
    verdict = ev.get('verdict') or {'0': 'PASS', '1': 'FAIL'}.get(rc, 'ERROR')
    label = 'calm' if cond == 'calm' else f'{wind} g{gust}'
    rows.append([case, label, verdict, fmt(m.get('final_dist_goal_m'), 2),
                 fmt(m.get('min_clearance_m'), 2), fmt(m.get('airborne_s')),
                 fmt(m.get('path_ratio'), 2), fmt(m.get('holds')),
                 fmt(tilt[0]) if tilt else '-', fmt(tilt[1]) if tilt else '-', folder])
    for text in ev.get('failures', []):
        notes.append(f'{case} {label}: {text}')
    if verdict == 'ERROR':
        notes.append(f'{case} {label}: setup error, see the trial log in the suite folder')

widths = [max(len(r[k]) for r in [header] + rows) for k in range(len(header))]
lines = ['  '.join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in [header] + rows]
lines.insert(1, '  '.join('-' * w for w in widths))
lines += [''] + (notes or ['no failures'])
lines += ['', 'goal m: final distance to the goal; clear m: nearest depth point in flight; '
          'path: flown / straight distance; tilt: deg in the air (PX4 ULog)',
          'Plots: path.png and timeseries.png in each flight folder.']
text = '\n'.join(lines)
print('\n' + text)
with open(out, 'w') as f:
    f.write(text + '\n')
print(f'\nSummary saved to {out}')
EOF

[ "$stop" = 0 ] || exit 130
worst=0
for r in "${results[@]}"; do
    rc=${r#*|*|}
    rc=${rc%%|*}
    case $rc in
        0) ;;
        1) [ "$worst" = 2 ] || worst=1 ;;
        *) worst=2 ;;
    esac
done
exit "$worst"
