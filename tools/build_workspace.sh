#!/usr/bin/env bash
# Build the same source on the simulation computer and the Orin.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/build_workspace.sh [--ros-distro humble|jazzy] [--jobs N] [--with-agent]

Build all ROS packages, using separate output directories for each ROS distro
and CPU architecture. Gazebo worlds/models are runtime assets, not colcon targets;
the simulation computer and Orin use the same ROS source and build command.

  --ros-distro NAME  Default: ROS_DISTRO, otherwise Humble on Ubuntu 22.04 or
                    Jazzy on Ubuntu 24.04. Requires /opt/ros/NAME/setup.bash.
  --jobs N          Compiler jobs (default: 2); packages build sequentially.
  --with-agent      Also build the copied Micro-XRCE-DDS-Agent 2.4.2 separately.
                    Humble: use installed FastCDR/FastDDS/spdlog, without downloads.
                    Jazzy: download pinned FastCDR 1.1.1, FastDDS 2.12.2 and
                    foonathan memory 0.7-3 into build/, then install privately.
                    This Jazzy compatibility path has not been verified on Orin;
                    PX4's native Jazzy recommendation is Agent 2.4.3.
  -h, --help        Show this help without building.

Install system/ROS dependencies first. This script does not install packages,
change source configuration, or start hardware, Gazebo, PX4, or flight missions.
Use a fresh terminal; do not source the workspace's generated install before building.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

distro=${ROS_DISTRO:-}
jobs=2
with_agent=false
while (($#)); do
    case "$1" in
        --ros-distro|--jobs)
            (($# >= 2)) || die "$1 requires a value."
            if [[ $1 == --ros-distro ]]; then distro=$2; else jobs=$2; fi
            shift 2 ;;
        --with-agent) with_agent=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)." ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || die '--jobs must be a positive integer.'
if [[ -z $distro ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    case ${VERSION_ID:-} in
        22.04) distro=humble ;;
        24.04) distro=jazzy ;;
        *) die 'Specify --ros-distro humble or --ros-distro jazzy.' ;;
    esac
fi
case $distro in humble|jazzy) ;; *) die "Unsupported ROS distro: $distro." ;; esac
[[ -z ${ROS_DISTRO:-} || $ROS_DISTRO == "$distro" ]] ||
    die "ROS $ROS_DISTRO is already sourced. Open a fresh terminal for $distro."
[[ -f /opt/ros/$distro/setup.bash ]] || die "ROS $distro is not installed in /opt/ros."
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$workspace"
[[ ! -L src/px4_control ]] || die 'Materialize src/px4_control before building this portable workspace.'
architecture=$(uname -m)
case $architecture in x86_64|aarch64) ;; *) die "Unsupported architecture: $architecture." ;; esac
output_tag=$distro-$architecture
if $with_agent; then
    [[ -f src/Micro-XRCE-DDS-Agent/CMakeLists.txt ]] || die 'Micro-XRCE-DDS-Agent source is missing.'
    command -v cmake >/dev/null || die 'Install cmake first.'
    if [[ $distro == jazzy ]]; then command -v git >/dev/null || die 'Install git first.'; fi
fi

# Source only the base ROS installation, never an existing workspace overlay.
# ROS setup scripts do not support nounset in all distributions.
# shellcheck disable=SC1090
source "/opt/ros/$distro/setup.bash"
set -u
command -v colcon >/dev/null || die 'Install colcon first.'
export MAKEFLAGS="-j$jobs" CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
colcon --log-base "log/$output_tag" build --base-paths src \
    --build-base "build/$output_tag" --install-base "install/$output_tag" \
    --symlink-install --executor sequential --packages-skip microxrcedds_agent \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
printf '\nROS packages built. In a new terminal, source %q\n' "$workspace/install/$output_tag/setup.bash"

if ! $with_agent; then exit 0; fi
agent_build="$workspace/build/agent-$output_tag"
agent_prefix="$workspace/install/agent-$output_tag"
agent_options=(-DUAGENT_SUPERBUILD=OFF -DUAGENT_P2P_PROFILE=OFF
    -DUAGENT_BUILD_TESTS=OFF -DUAGENT_USE_SYSTEM_FASTCDR=ON
    -DUAGENT_USE_SYSTEM_FASTDDS=ON -DUAGENT_USE_SYSTEM_LOGGER=ON)

build_install() {
    local source_dir=$1 build_dir=$2
    shift 2
    cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$agent_prefix" \
        "-DCMAKE_INSTALL_RPATH=$agent_prefix/lib;$agent_prefix/lib64" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF "$@"
    cmake --build "$build_dir" --parallel "$jobs"
    cmake --install "$build_dir"
}

dependency() {
    local name=$1 url=$2 tag=$3
    shift 3
    local source_dir="$agent_build/deps/src/$name"
    if [[ ! -e $source_dir ]]; then
        mkdir -p "$agent_build/deps/src"
        git clone --depth 1 --branch "$tag" "$url" "$source_dir"
    fi
    [[ $(git -C "$source_dir" rev-parse HEAD) == \
       "$(git -C "$source_dir" rev-parse "$tag^{commit}")" ]] ||
        die "Dependency $source_dir is not at $tag."
    git -C "$source_dir" diff --quiet HEAD || die "Dependency $source_dir has local edits."
    build_install "$source_dir" "$agent_build/deps/build/$name" \
        -DCMAKE_PREFIX_PATH="$agent_prefix" -DBUILD_SHARED_LIBS=ON "$@"
}

if [[ $distro == jazzy ]]; then
    printf '\nBuilding Agent 2.4.2 with private DDS libraries; downloads require network access.\n'
    printf 'The Jazzy/Orin path is not hardware verified. Native Jazzy normally uses Agent 2.4.3.\n'
    (
        # Keep FastCDR 1 and this Agent out of Jazzy's ROS library environment.
        unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH
        dependency fastcdr https://github.com/eProsima/Fast-CDR.git v1.1.1 -DBUILD_TESTING=OFF
        dependency foonathan_memory https://github.com/foonathan/memory.git v0.7-3 \
            -DFOONATHAN_MEMORY_BUILD_EXAMPLES=OFF -DFOONATHAN_MEMORY_BUILD_TESTS=OFF
        dependency fastdds https://github.com/eProsima/Fast-DDS.git v2.12.2 \
            -DTHIRDPARTY=OFF -DBUILD_TESTING=OFF -DCOMPILE_EXAMPLES=OFF \
            -DCOMPILE_TOOLS=OFF -DSECURITY=OFF -DSHM_TRANSPORT_DEFAULT=OFF
        build_install "$workspace/src/Micro-XRCE-DDS-Agent" "$agent_build/agent" \
            -DCMAKE_PREFIX_PATH="$agent_prefix" "${agent_options[@]}"
    )
else
    build_install "$workspace/src/Micro-XRCE-DDS-Agent" "$agent_build/agent" "${agent_options[@]}"
fi
printf '\nAgent installed at %q (not started).\n' "$agent_prefix/bin/MicroXRCEAgent"
