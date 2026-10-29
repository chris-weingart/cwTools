#!/usr/bin/env bash
# Launch Flame with cwFill discoverable from the dev bundle, with
# the OFX wrapper's debug output enabled (Philippe Jean's env vars) and the
# plugin cache disabled so rebuilds are picked up immediately.
#
#   ./scripts/flame_dev_launch.sh [/opt/Autodesk/flame_2027.prNNN]
#
# Defaults to the newest installed flame_2027.* . Run scripts/build.sh first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FLAME_DIR="${1:-$(ls -d /opt/Autodesk/flame_2027.* 2>/dev/null | sort | tail -1)}"
START="$FLAME_DIR/bin/startApplication"
[[ -x "$START" ]] || { echo "no startApplication at $START" >&2; exit 1; }
[[ -d "$ROOT/dist/cwFill.ofx.bundle" ]] || { echo "build first: scripts/build.sh" >&2; exit 1; }

# OFX_PLUGIN_PATH is searched in ADDITION to the system dirs.
export OFX_PLUGIN_PATH="$ROOT/dist"
export DL_OFX_PLUGIN_CACHE_ENABLED=0
export OFX_DEBUG_ACTIONS=1
export OFX_DEBUG_PARAMETERS=1
# export OFX_DEBUG_ALL=1   # uncomment for the full firehose

echo "Flame:           $FLAME_DIR"
echo "OFX_PLUGIN_PATH: $OFX_PLUGIN_PATH"
echo "Look for the 'cw' group / 'cwFill' node in a Batch schematic."
echo "------------------------------------------------------------------"
exec "$START"
