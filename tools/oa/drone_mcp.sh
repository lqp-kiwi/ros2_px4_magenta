#!/usr/bin/env bash
# Launch the read-only "drone" MCP server (stdio) with ROS 2 and the workspace install sourced.
# Nothing may be printed on stdout except the protocol, so the sourcing output is discarded.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh" > /dev/null 2>&1
exec python3 -u "$HERE/drone_mcp.py" "$@"
