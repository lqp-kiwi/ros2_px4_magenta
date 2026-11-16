#!/usr/bin/env bash
# Headless PX4 SITL + Gazebo stack for automated tests, isolated from the user's own stack.
#
#   tools/oa/sitl.sh up [WORLD] [--wind E,N [--gust G]]
#                                 start gz (server only), PX4, MicroXRCEAgent and the depth bridge
#   tools/oa/sitl.sh down         stop what `up` started, plus a hung gz in its partition
#   tools/oa/sitl.sh status       our processes and any other simulator processes
#   tools/oa/sitl.sh env          export lines for the isolated stack: eval "$(tools/oa/sitl.sh env)"
#
# WORLD is worlds/<WORLD>.sdf of this workspace (default my_forest).
# --wind E,N: Gazebo runs a copy of the world with wind, E m/s towards east and N m/s towards north
# (world x, y); --gust G (default 0.3) varies its speed by +-G, 0 = constant. tools/oa/wind_world.py
# writes the copy to $OA_STATE/windy/; the world files are not changed.
# Isolation: GZ_PARTITION=drone_oa, ROS_DOMAIN_ID=$OA_DOMAIN_ID (default 42; PX4's rcS copies it to
# UXRCE_DDS_DOM_ID), own PX4 working directory $OA_STATE/px4 with a fresh copy of the user's
# parameters.bson (SIM_GZ_HOME_LAT/LON, without it the GPS goals land elsewhere).
# PX4 instance 0 ports (uXRCE 8888, MAVLink 14580) are shared with the user's stack, so `up` refuses
# while any other gz sim / px4 / MicroXRCEAgent / bridge / mavproxy process runs.
# This script never starts a px4_control mission; tools/oa/run_trial.sh does.
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot-1.14}"
OA_STATE="${OA_STATE:-${XDG_CACHE_HOME:-$HOME/.cache}/drone-oa/sitl}"
OA_DOMAIN_ID="${OA_DOMAIN_ID:-42}"
PARTITION=drone_oa
PIDS="$OA_STATE/pids"

die() { echo "sitl.sh: $*" >&2; exit 2; }

iso_env() {
    export GZ_PARTITION="$PARTITION"
    export ROS_DOMAIN_ID="$OA_DOMAIN_ID"
    export GZ_SIM_RESOURCE_PATH="$WS/models:$HOME/ros2_px4_gz_ws/models:$PX4_DIR/Tools/simulation/gz/models"
    export GZ_SIM_SERVER_CONFIG_PATH="$PX4_DIR/src/modules/simulation/gz_bridge/server.config"
}

our_pids() { [ -f "$PIDS" ] && awk '{print $2}' "$PIDS" || true; }

alive() { kill -0 "$1" 2>/dev/null; }

# PIDs of every process in our process groups (each started with setsid: pgid = recorded pid)
our_tree() {
    local p
    for p in $(our_pids); do pgrep -g "$p" 2>/dev/null || true; done
}

# Simulator-related processes that are not ours
foreign() {
    local mine p cmd
    mine=" $(our_tree | tr '\n' ' ') "
    for p in $(pgrep -f 'gz sim|bin/px4( |$)|MicroXRCEAgent|parameter_bridge|mavproxy|mavlink-routerd' || true); do
        [ "$p" = "$$" ] && continue
        case "$mine" in *" $p "*) continue ;; esac
        cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$cmd" in *sitl.sh*|*pgrep*|"") continue ;; esac
        echo "$p ${cmd:0:120}"
    done
}

start() {   # name, log file, command...
    local name=$1 log=$2
    shift 2
    setsid "$@" > "$log" 2>&1 < /dev/null &
    echo "$name $!" >> "$PIDS"
}

wait_for() {   # seconds, description, test command...
    local secs=$1 what=$2 i
    shift 2
    for ((i = 0; i < secs; i++)); do
        if "$@" > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    die "timed out after ${secs}s waiting for $what (logs in $OA_STATE)"
}

cmd_up() {
    local world=my_forest wind="" gust=0.3 sdf
    while [ $# -gt 0 ]; do
        case $1 in
            --wind) [ $# -ge 2 ] || die "--wind needs E,N"; wind=$2; shift 2 ;;
            --gust) [ $# -ge 2 ] || die "--gust needs a fraction"; gust=$2; shift 2 ;;
            -*) die "unknown option $1" ;;
            *) world=$1; shift ;;
        esac
    done
    sdf="$WS/worlds/$world.sdf"
    [ -f "$sdf" ] || die "no world file $sdf"
    [ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ] || die "PX4 SITL not built in $PX4_DIR"
    if [ -n "$(our_tree)" ]; then
        echo "sitl.sh: already up (world $(cat "$OA_STATE/world" 2>/dev/null), wind $(wind_setting))"
        return 0
    fi
    local others
    others=$(foreign)
    [ -z "$others" ] || die "other simulator processes are running (they share PX4 instance 0 ports); stop them first:
$others"

    mkdir -p "$OA_STATE/px4"
    : > "$PIDS"
    iso_env
    rm -rf "$OA_STATE/windy"
    if [ -n "$wind" ]; then
        mkdir -p "$OA_STATE/windy"
        python3 "$WS/tools/oa/wind_world.py" "$sdf" "$OA_STATE/windy/$world.sdf" --wind "$wind" \
            --gust "$gust" --server-config "$GZ_SIM_SERVER_CONFIG_PATH" "$OA_STATE/windy/server.config" ||
            die "could not write the windy world"
        sdf="$OA_STATE/windy/$world.sdf"
        export GZ_SIM_SERVER_CONFIG_PATH="$OA_STATE/windy/server.config"
    fi
    echo "$world" > "$OA_STATE/world"
    echo "${wind:+$wind gust $gust}" > "$OA_STATE/wind"
    # shellcheck disable=SC1091
    source "$WS/tools/oa/env.sh" > /dev/null

    local gz_args=(sim -s -r "$sdf")
    [ -n "${DISPLAY:-}" ] || gz_args=(sim -s -r --headless-rendering "$sdf")
    start gz "$OA_STATE/gz.log" gz "${gz_args[@]}"
    wait_for 90 "the Gazebo world $world" bash -c "gz topic -l | grep -q '^/world/$world/clock'"

    # fresh parameters each start, PX4 logs (ULog) kept in $OA_STATE/px4/log
    rm -f "$OA_STATE/px4/parameters.bson" "$OA_STATE/px4/parameters_backup.bson"
    if [ -f "$PX4_DIR/build/px4_sitl_default/rootfs/parameters.bson" ]; then
        cp "$PX4_DIR/build/px4_sitl_default/rootfs/parameters.bson" "$OA_STATE/px4/"
    else
        echo "sitl.sh: warning: no parameters.bson to copy, GPS home is PX4's default" >&2
    fi
    pgrep -f 'bin/px4( |$)' > /dev/null || rm -f /tmp/px4-sock-0 /tmp/px4_lock-0
    start agent "$OA_STATE/agent.log" MicroXRCEAgent udp4 -p 8888
    (cd "$OA_STATE/px4" && export PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=gz_x500_depth_down &&
        start px4 "$OA_STATE/px4.log" "$PX4_DIR/build/px4_sitl_default/bin/px4" -d \
        -w "$OA_STATE/px4" "$PX4_DIR/build/px4_sitl_default/etc")
    start bridge "$OA_STATE/bridge.log" ros2 run ros_gz_bridge parameter_bridge \
        "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image" \
        "/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
        --ros-args -r /depth_camera:=/camera/depth/image_raw -r /camera_info:=/camera/depth/camera_info
    wait_for 150 "PX4 'Ready for takeoff'" grep -q 'Ready for takeoff' "$OA_STATE/px4.log"
    echo "sitl.sh: up: world $world, wind $(wind_setting), GZ_PARTITION=$PARTITION," \
        "ROS_DOMAIN_ID=$OA_DOMAIN_ID, logs in $OA_STATE"
}

# Wind of the running (or last) stack: "E,N gust G", or "none"
wind_setting() {
    local w
    w=$(cat "$OA_STATE/wind" 2>/dev/null || true)
    echo "${w:-none}"
}

cmd_down() {
    local p i
    for p in $(our_pids); do kill -INT -- "-$p" 2>/dev/null || true; done
    for ((i = 0; i < 10; i++)); do
        [ -z "$(our_tree)" ] && break
        sleep 1
    done
    for p in $(our_pids); do kill -TERM -- "-$p" 2>/dev/null || true; done
    sleep 1
    for p in $(our_tree); do kill -KILL "$p" 2>/dev/null || true; done
    # a headless gz can hang while shutting down and keep advertising its services
    for p in $(pgrep -f 'gz sim' || true); do
        if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "GZ_PARTITION=$PARTITION"; then
            kill -KILL "$p" 2>/dev/null || true
        fi
    done
    pgrep -f 'bin/px4( |$)' > /dev/null || rm -f /tmp/px4-sock-0 /tmp/px4_lock-0
    rm -f "$PIDS"
    echo "sitl.sh: down"
}

cmd_status() {
    local line name p
    if [ -f "$PIDS" ]; then
        echo "ours (world $(cat "$OA_STATE/world" 2>/dev/null), wind $(wind_setting)," \
            "ROS_DOMAIN_ID=$OA_DOMAIN_ID, GZ_PARTITION=$PARTITION):"
        while read -r name p; do
            if alive "$p"; then echo "  $name  pid $p  running"; else echo "  $name  pid $p  exited"; fi
        done < "$PIDS"
        grep -q 'Ready for takeoff' "$OA_STATE/px4.log" 2>/dev/null && echo "  PX4 ready for takeoff"
    else
        echo "ours: not running"
    fi
    line=$(foreign)
    if [ -n "$line" ]; then echo "other simulator processes:"; echo "$line" | sed 's/^/  /'; fi
}

cmd_env() {
    echo "export GZ_PARTITION=$PARTITION ROS_DOMAIN_ID=$OA_DOMAIN_ID"
}

case "${1:-status}" in
    up) shift; cmd_up "$@" ;;
    down) cmd_down ;;
    status) cmd_status ;;
    env) cmd_env ;;
    -h|--help|help) awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0" ;;
    *) die "unknown command '$1' (up|down|status|env)" ;;
esac
