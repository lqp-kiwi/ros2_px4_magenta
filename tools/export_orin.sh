#!/usr/bin/env bash
# Export portable sources without workspace build products or Git metadata.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/export_orin.sh [OUTPUT.tar.gz]

Export src and any drivers.repos, tools, docs, and README.md beside it.
Symlinks are copied as their target contents so the archive is portable.
Git metadata, build products, logs, flight data, and Python caches are excluded.

The default output is /tmp/ros2_px4_orin_<timestamp>.tar.gz.
The destination must be outside the exported inputs and must not exist.
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
[[ $# -le 1 ]] || fail 'Expected at most one output path; see --help.'
[[ "${1:-}" != -* ]] || fail 'Unknown option; see --help.'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "$script_dir/.." && pwd)"
[[ -d "$workspace_dir/src" ]] || fail 'Cannot find workspace src directory.'

output="${1:-/tmp/ros2_px4_orin_$(date +%Y%m%d_%H%M%S).tar.gz}"
[[ ! -e "$output" && ! -L "$output" ]] || fail "Output already exists: $output"
output="$(realpath -m -- "$output")"
[[ -d "$(dirname -- "$output")" ]] || fail 'Output directory does not exist.'

inputs=(src)
for optional in drivers.repos tools docs README.md; do
    if [[ -e "$workspace_dir/$optional" ]]; then
        inputs+=("$optional")
    fi
done

for input in "${inputs[@]}"; do
    input_path="$(realpath -m -- "$workspace_dir/$input")"
    case "$output" in
        "$input_path"|"$input_path"/*)
            fail "Output must be outside exported input: $input"
            ;;
    esac
done

# Reserve the destination without overwriting it, including in a concurrent run.
if ! (set -o noclobber; : > "$output") 2>/dev/null; then
    fail "Cannot create output, or it already exists: $output"
fi
trap 'rm -f -- "$output"' EXIT

tar --create --gzip --dereference --file="$output" \
    --exclude-vcs \
    --exclude=.git \
    --exclude=build --exclude=install --exclude=log --exclude=flight_logs \
    --exclude=__pycache__ --exclude='*.pyc' --exclude='*.pyo' \
    --exclude='*.egg-info' --exclude=.pytest_cache \
    --directory="$workspace_dir" -- "${inputs[@]}"

trap - EXIT
printf 'Exported sources to %s\n' "$output"
