# shellcheck shell=bash
# Source ROS 2 and this workspace's install. Usage: source tools/oa/env.sh
# Prefers the per-distro layout of tools/build_workspace.sh (install/<distro>-<arch>),
# falls back to the legacy install/setup.bash. Sets OA_WS and OA_INSTALL.
OA_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -z "${ROS_DISTRO:-}" ]; then
    for d in humble jazzy; do
        [ -f "/opt/ros/$d/setup.bash" ] && { ROS_DISTRO=$d; break; }
    done
fi
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
OA_INSTALL="$OA_WS/install/${ROS_DISTRO}-$(uname -m)"
[ -f "$OA_INSTALL/setup.bash" ] || OA_INSTALL="$OA_WS/install"
if [ -f "$OA_INSTALL/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$OA_INSTALL/setup.bash"
else
    echo "env.sh: no workspace install found; build first (bash tools/build_workspace.sh)" >&2
fi
export OA_WS OA_INSTALL
