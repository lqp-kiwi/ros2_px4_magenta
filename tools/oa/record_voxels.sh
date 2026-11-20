#!/usr/bin/env bash
# Record the 3DVFH+ point memory of the real depth camera until you press Enter, then make the
# voxel video (2D, voxels only). Camera only: no PX4 link, nothing is published, nothing moves.
#
#   tools/oa/record_voxels.sh [--height M] [--time S]     from anywhere, on the Orin (or the PC)
#
# 1. Starts the depth camera (ros2 run depth_camera camera --no-cloud) unless it already runs.
# 2. Runs tools/oa/depth_filter_check.py (the node's depth filters + the stability check of
#    vfh3d.py) until you stop it: Enter, or Ctrl+C, or from another terminal
#    `pkill -TERM -f depth_filter_check`. --time S stops by itself after S seconds.
# 3. Stops the camera it started and renders voxel_video.mp4 with tools/oa/voxel_video.py.
# Output: <workspace>/depth_checks/<date>_<time>/ (vfh.npz, check.txt, camera.log,
# voxel_video.mp4). No PX4 pose: the drone is taken as still, level and facing north, so moving
# it smears the voxels. --height M: camera height above the floor (default 0.3 m).
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WS"
case "$1" in
    -h|--help) sed -n '2,/^set -eo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
esac

# ROS 2: Jazzy on the Orin, Humble on the PC. Workspace: the PC's install/<distro>-<arch>, else the
# Orin's plain install/ (on the PC install/setup.bash is an old copy: never use it there).
for distro in jazzy humble; do
    if [ -f "/opt/ros/$distro/setup.bash" ]; then
        # shellcheck source=/dev/null
        source "/opt/ros/$distro/setup.bash"
        break
    fi
done
if [ -f "install/$ROS_DISTRO-$(uname -m)/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "install/$ROS_DISTRO-$(uname -m)/setup.bash"
else
    # shellcheck source=/dev/null
    source install/setup.bash
fi
python3 -c "import px4_control.vfh3d" 2> /dev/null ||
    { echo "px4_control is not built in $WS" >&2; exit 1; }

RUN=$(mktemp -d /tmp/record_voxels_XXXXXX)
camera_pg=""
stop_group() {    # process group, first signal; then TERM, then KILL for what is still running
    local pg=$1 sig pid
    [ -n "$pg" ] || return 0
    for sig in "$2" TERM KILL; do
        kill -"$sig" -- "-$pg" 2> /dev/null || return 0
        for pid in $(pgrep -g "$pg"); do timeout 8 tail --pid="$pid" -f /dev/null || true; done
    done
}
cleanup() {
    stop_group "$camera_pg" INT     # the camera driver shuts down on SIGINT
    camera_pg=""
}
trap cleanup EXIT

if pgrep -f "orbbec_camera_node" > /dev/null; then
    echo "-- Depth camera already running, using it"
else
    setsid ros2 run depth_camera camera --no-cloud > "$RUN/camera.log" 2>&1 < /dev/null &
    camera_pg=$!
    echo "-- Depth camera starting (log $RUN/camera.log)"
fi
if ! timeout 30 ros2 topic echo --once --qos-reliability best_effort /camera/depth/camera_info \
        sensor_msgs/msg/CameraInfo > /dev/null 2>&1; then
    echo "No depth camera data in 30 s" >&2
    tail -n 5 "$RUN/camera.log" >&2 2> /dev/null || true
    exit 1
fi
echo "-- Camera OK"

mkdir -p depth_checks
touch "$RUN/start"
trap ':' INT    # Ctrl+C goes to the recorder (it saves its log); this script then carries on
rc=0
python3 tools/oa/depth_filter_check.py --time 0 --out depth_checks "$@" || rc=$?
trap - INT

dir=$(find depth_checks -mindepth 1 -maxdepth 1 -type d -newer "$RUN/start" | sort | tail -1)
cleanup
if [ -z "$dir" ] || [ ! -f "$dir/vfh.npz" ]; then
    echo "The recorder wrote no vfh.npz (exit code $rc)" >&2
    exit 1
fi
[ -f "$RUN/camera.log" ] && cp "$RUN/camera.log" "$dir/"
echo "-- Making the video..."
if python3 tools/oa/voxel_video.py "$dir"; then
    echo "-- Done: $WS/$dir/voxel_video.mp4"
else
    echo "Video failed here (needs PIL and gst-launch-1.0 with x264enc): copy $WS/$dir to the PC" \
         "and run python3 tools/oa/voxel_video.py <folder>" >&2
    exit 1
fi
