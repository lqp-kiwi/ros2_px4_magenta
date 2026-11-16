#!/usr/bin/env bash
# Gazebo with its GUI for the person's own SITL stack (terminal 1, instead of ~/make_world.sh):
# any world of this workspace, optionally with wind.
#
#   tools/oa/gui_world.sh [WORLD] [--wind E,N [--gust G]]
#
# WORLD is worlds/<WORLD>.sdf of this workspace: my_forest (default), grass_field, my_world.
# --wind E,N: the air moves E m/s towards east and N m/s towards north; --gust G (default 0.3)
# varies its speed by +-G, 0 = constant. Same wind as tools/oa/sitl.sh (tools/oa/wind_world.py),
# the windy copy is written to ~/.cache/drone-oa/gui/.
# Then the other terminals as usual (AGENTS.md, Running): PX4, MicroXRCEAgent udp4 -p 8888, the
# depth bridge and the node with the goal of the world. Refuses while another simulator runs
# (the headless stack or a wind suite included): they would share PX4 instance 0.
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot-1.14}"
OUT="${XDG_CACHE_HOME:-$HOME/.cache}/drone-oa/gui"
world=my_forest
wind=""
gust=0.3

die() { echo "gui_world.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case $1 in
        --wind) [ $# -ge 2 ] || die "--wind needs E,N"; wind=$2; shift 2 ;;
        --gust) [ $# -ge 2 ] || die "--gust needs a fraction"; gust=$2; shift 2 ;;
        -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
        -*) die "unknown option $1" ;;
        *) world=$1; shift ;;
    esac
done

sdf="$WS/worlds/$world.sdf"
if [ ! -f "$sdf" ]; then
    names=$(for f in "$WS"/worlds/*.sdf; do basename "$f" .sdf; done | tr '\n' ' ')
    die "no world file $sdf (worlds: $names)"
fi
status=$("$WS/tools/oa/sitl.sh" status)
if grep -qE '^  [a-z]+  pid [0-9]+  running$|^other simulator' <<< "$status"; then
    echo "$status" >&2
    die "a simulator is running: wait for it to finish or stop it first (tools/oa/sitl.sh down)"
fi

export GZ_SIM_RESOURCE_PATH="$WS/models:$HOME/ros2_px4_gz_ws/models:$PX4_DIR/Tools/simulation/gz/models"
export GZ_SIM_SERVER_CONFIG_PATH="$PX4_DIR/src/modules/simulation/gz_bridge/server.config"
unset GZ_PARTITION      # PX4 in the other terminal looks for the world in the default partition
if [ -n "$wind" ]; then
    mkdir -p "$OUT"
    python3 "$WS/tools/oa/wind_world.py" "$sdf" "$OUT/$world.sdf" --wind "$wind" --gust "$gust" \
        --server-config "$GZ_SIM_SERVER_CONFIG_PATH" "$OUT/server.config"
    sdf="$OUT/$world.sdf"
    export GZ_SIM_SERVER_CONFIG_PATH="$OUT/server.config"
fi
echo "gui_world.sh: world $world, wind ${wind:-none}${wind:+ m/s, gust $gust}"
exec gz sim -r "$sdf"
