#!/usr/bin/env bash
# Install the built bundle into the system-wide OFX plugin directory so EVERY
# OFX host on the machine (all Flame/Flare installations, Resolve, Nuke, ...)
# lists it. Copies the dist bundle (a real dir, same pattern as the installed
# abToolsDefocus / Neat Video bundles) — re-run after each build.sh.
#
# Needs root to write the system dir, so run via sudo:
#   sudo ./scripts/install.sh         (or  ! ./scripts/install.sh  in Claude Code)
#
# Ships BOTH in the repo (scripts/install.sh, bundle at ../dist/) and at the top
# level of the release archive (bundle right next to this script) — it finds the
# bundle in either layout, so `sudo ./install.sh` works straight out of the tarball.
#
# Dev alternative (no sudo, per-launch): scripts/flame_dev_launch.sh, which sets
# OFX_PLUGIN_PATH for a single Flame session.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Release-archive layout first (bundle beside the script), then repo layout.
for CANDIDATE in "$HERE/cwFill.ofx.bundle" "$HERE/../dist/cwFill.ofx.bundle"; do
  [[ -d "$CANDIDATE" ]] && { BUNDLE="$(cd "$CANDIDATE" && pwd)"; break; }
done
[[ -n "${BUNDLE:-}" ]] || {
  echo "no cwFill.ofx.bundle next to this script or in ../dist — in the repo, run scripts/build.sh first" >&2
  exit 1
}

case "$(uname -s)" in
  Darwin) DEST="/Library/OFX/Plugins" ;;
  Linux)  DEST="/usr/OFX/Plugins" ;;
  *) echo "unsupported OS" >&2; exit 1 ;;
esac
# Test/staging escape hatch — normal installs never set this.
DEST="${CWFILL_DEST:-$DEST}"

# Copy (not symlink) — a real bundle dir matches the existing abToolsDefocus /
# NeatVideo installs and survives dist rebuilds. For live dev iteration on the
# build machine, prefer scripts/flame_dev_launch.sh (OFX_PLUGIN_PATH) instead.
mkdir -p "$DEST"
TARGET="$DEST/cwFill.ofx.bundle"
rm -rf "$TARGET"
cp -R "$BUNDLE" "$TARGET"
echo "installed (copy): $TARGET"

echo "OFX hosts that will now list it:"
ls -d /opt/Autodesk/flame_2027.* /opt/Autodesk/flare_* 2>/dev/null | sed 's/^/  - /' || true
echo "Stale params after a rebuild? delete ofxPluginCache.xml or launch with DL_OFX_PLUGIN_CACHE_ENABLED=0"
