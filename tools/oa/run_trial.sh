#!/usr/bin/env bash
# One automated SITL trial: start the isolated headless stack (tools/oa/sitl.sh), fly a px4_control
# node to a GPS goal, score its flight log with tools/oa/flight_eval.py.
#
#   tools/oa/run_trial.sh [--node NAME] [--world NAME] [--goal FILE] [--wind E,N [--gust G]]
#                         [--timeout S] [--keep-up]
#
# Defaults: --node obstacle_avoidance_static_offboard --world my_forest --goal worlds/forest_goal.json,
#           no wind, --timeout 900 (then Ctrl+C: the node switches PX4 to Hold and exits).
# --wind E,N / --gust G: wind in Gazebo, see tools/oa/sitl.sh. A headless stack that is already up
# must have the same world and wind, else the trial refuses to start.
# The node gets Enter on stdin for its "Enter to take off" prompt and runs from the workspace root,
# so its log lands in flight_logs/<date>_<time>/. The PX4 ULog of the trial is copied there as
# px4.ulg, the score as eval.json, the trial settings as trial.json. Trial output:
# ~/.cache/drone-oa/trials/<date>_<time>/. Ctrl+C is passed on to the node: Hold, log saved.
# Exit code: 0 PASS, 1 FAIL, 2 setup error. Runs only on the headless stack, never on a vehicle.
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NODE=obstacle_avoidance_static_offboard
WORLD=my_forest
GOAL=worlds/forest_goal.json
WIND=""
GUST=0.3
TIMEOUT=900
KEEP_UP=0
while [ $# -gt 0 ]; do
    case $1 in
        --node) NODE=$2; shift 2 ;;
        --world) WORLD=$2; shift 2 ;;
        --goal) GOAL=$2; shift 2 ;;
        --wind) WIND=$2; shift 2 ;;
        --gust) GUST=$2; shift 2 ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        --keep-up) KEEP_UP=1; shift ;;
        -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
        *) echo "run_trial.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

cd "$WS"
[ -f "$GOAL" ] || { echo "run_trial.sh: no goal file $GOAL" >&2; exit 2; }
# shellcheck disable=SC1091
source tools/oa/env.sh > /dev/null
ros2 pkg executables px4_control 2>/dev/null | grep -qx "px4_control $NODE" ||
    { echo "run_trial.sh: px4_control has no executable $NODE (build first)" >&2; exit 2; }

# A copied (non --symlink-install) package runs stale code after an edit
prefix=$(ros2 pkg prefix px4_control)
installed=$(ls -d "$prefix"/lib/python3*/site-packages/px4_control 2>/dev/null | head -1 || true)
if [ -n "$installed" ] && [ ! -L "$installed" ]; then
    for f in src/px4_control/px4_control/*.py; do
        if ! cmp -s "$f" "$installed/$(basename "$f")"; then
            echo "run_trial.sh: $installed is an outdated copy of $f: rebuild px4_control first" >&2
            exit 2
        fi
    done
fi

stamp=$(date +%Y%m%d_%H%M%S)
TRIAL="${XDG_CACHE_HOME:-$HOME/.cache}/drone-oa/trials/$stamp"
mkdir -p "$TRIAL" flight_logs
LOCK=flight_logs/.trial_running
started_stack=0
cleanup() {
    rm -f "$LOCK"
    if [ "$started_stack" = 1 ] && [ "$KEEP_UP" = 0 ]; then tools/oa/sitl.sh down > /dev/null || true; fi
}
trap cleanup EXIT

OA_STATE="${OA_STATE:-${XDG_CACHE_HOME:-$HOME/.cache}/drone-oa/sitl}"
want_wind="${WIND:+$WIND gust $GUST}"
stack_status=$(tools/oa/sitl.sh status)
if grep -q 'PX4 ready' <<< "$stack_status"; then
    have_world=$(cat "$OA_STATE/world" 2>/dev/null || true)
    have_wind=$(cat "$OA_STATE/wind" 2>/dev/null || true)
    if [ "$have_world" != "$WORLD" ] || [ "$have_wind" != "$want_wind" ]; then
        echo "run_trial.sh: the headless stack is up with world $have_world, wind ${have_wind:-none};" \
            "this trial wants $WORLD, wind ${want_wind:-none}. Stop it first: tools/oa/sitl.sh down" >&2
        exit 2
    fi
else
    up_args=("$WORLD")
    if [ -n "$WIND" ]; then up_args+=(--wind "$WIND" --gust "$GUST"); fi
    started_stack=1     # set before `up`: if it fails half way, cleanup stops what it started
    tools/oa/sitl.sh up "${up_args[@]}" | tee "$TRIAL/sitl.txt"
fi
eval "$(tools/oa/sitl.sh env)"
echo "$$ $stamp $NODE $WORLD" > "$LOCK"

before=$(cd flight_logs && printf '%s\n' */)
echo "run_trial.sh: flying $NODE to $GOAL in $WORLD, wind ${want_wind:-none} (timeout ${TIMEOUT}s," \
    "node log $TRIAL/node.log)"
# timeout runs the node in its own process group, which Ctrl+C in this terminal does not reach:
# pass Ctrl+C on to it, the node then switches PX4 to Hold and saves its log
printf '\n' | timeout --signal=INT --kill-after=30 "$TIMEOUT" ros2 run px4_control "$NODE" "$GOAL" \
    > "$TRIAL/node.log" 2>&1 &
flight=$!
trap 'kill -INT "$flight" 2>/dev/null || true' INT TERM
node_rc=""
while [ -z "$node_rc" ]; do
    rc=0
    wait "$flight" || rc=$?
    kill -0 "$flight" 2>/dev/null || node_rc=$rc
done
trap - INT TERM
echo "run_trial.sh: node exited with code $node_rc$([ "$node_rc" = 124 ] && echo ' (timeout)')"

folder=""
for d in flight_logs/*/; do
    d=${d#flight_logs/}
    if ! grep -qxF "$d" <<< "$before"; then folder="flight_logs/${d%/}"; fi
done
if [ -z "$folder" ] || [ ! -f "$folder/flight.csv" ]; then
    echo "run_trial.sh: the node wrote no flight.csv; last lines of $TRIAL/node.log:" >&2
    tail -20 "$TRIAL/node.log" >&2
    exit 2
fi
cp "$TRIAL/node.log" "$folder/node.log"
python3 - "$folder/trial.json" "$NODE" "$WORLD" "$GOAL" "$WIND" "$GUST" "$node_rc" "$TRIAL" <<'EOF'
import json
import sys
path, node, world, goal, wind, gust, rc, trial = sys.argv[1:]
with open(path, 'w') as f:
    json.dump({'node': node, 'world': world, 'goal': goal,
               'wind_east_north_mps': [float(v) for v in wind.split(',')] if wind else None,
               'gust': float(gust) if wind else None, 'node_exit_code': int(rc),
               'trial_output': trial, 'stack': 'tools/oa/sitl.sh (headless SITL)'}, f, indent=2)
    f.write('\n')
EOF
ulog=$(find "${XDG_CACHE_HOME:-$HOME/.cache}/drone-oa/sitl/px4/log" -name '*.ulg' -newer "$TRIAL" \
    2>/dev/null | sort | tail -1)
[ -n "$ulog" ] && cp "$ulog" "$folder/px4.ulg"

set +e
python3 tools/oa/flight_eval.py "$folder" --world "worlds/$WORLD.sdf" --write
verdict_rc=$?
set -e
echo "run_trial.sh: flight folder $folder"
exit $verdict_rc
