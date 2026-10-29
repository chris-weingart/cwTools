#!/usr/bin/env bash
# Build cwFill and package it as an .ofx.bundle.
# Mirrors the abTools-Defocus packaging: Contents/<arch>/<name>.ofx
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Single source of truth for the version: Cargo.toml. (Hardcoding it in the
# plists is how CFBundleShortVersionString silently stayed 0.1.0 once.)
VERSION="$(grep -m1 '^version' crates/ofx-plugin/Cargo.toml | cut -d'"' -f2)"
[[ -n "$VERSION" ]] || { echo "could not read version from Cargo.toml" >&2; exit 1; }

# no build machine paths in the binaries (panic locations, debug info)
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$ROOT=/src"
cargo build --release

BUNDLE="dist/cwFill.ofx.bundle"
rm -rf "$BUNDLE"

case "$(uname -s)" in
  Linux)
    # Standard OFX layout: Contents/<arch>/<name>.ofx . Flame scans the arch
    # dir for *.ofx; the Info.plist is cosmetic on Linux but we write one to
    # match the known-good abToolsDefocus bundle exactly.
    DEST="$BUNDLE/Contents/Linux-x86-64"
    mkdir -p "$DEST"
    cp "target/release/libcwfill.so" "$DEST/cwFill.ofx"
    cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>cwFill</string>
    <key>CFBundlePackageType</key><string>BNDL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>CFBundleDevelopmentRegion</key><string>English</string>
    <key>CFBundleIdentifier</key><string>dev.weingart.cwFill</string>
</dict>
</plist>
PLIST
    ;;
  Darwin)
    # macOS uses the Apple bundle layout (verified against Neat Video's bundle
    # that Flame loads): Contents/MacOS/<name>.ofx + Info.plist + PkgInfo.
    DEST="$BUNDLE/Contents/MacOS"
    mkdir -p "$DEST"
    cp "target/release/libcwfill.dylib" "$DEST/cwFill.ofx"
    install_name_tool -id "@rpath/cwFill.ofx" "$DEST/cwFill.ofx"  # the linker id is the build path
    cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key><string>English</string>
	<key>CFBundleExecutable</key><string>cwFill.ofx</string>
	<key>CFBundleIdentifier</key><string>dev.weingart.cwFill</string>
	<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
	<key>CFBundleName</key><string>cwFill</string>
	<key>CFBundlePackageType</key><string>BNDL</string>
	<key>CFBundleShortVersionString</key><string>${VERSION}</string>
	<key>CFBundleSignature</key><string>????</string>
	<key>CFBundleVersion</key><string>${VERSION}</string>
	<key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
PLIST
    printf 'BNDL????' > "$BUNDLE/Contents/PkgInfo"
    # Ad-hoc codesign so Gatekeeper/hardened-runtime hosts will dlopen it.
    codesign --force --sign - "$DEST/cwFill.ofx" 2>/dev/null \
      && echo "ad-hoc signed" || echo "codesign skipped"
    ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

# Optional PNG icon (only resource type Flame reads).
if [[ -f "resources/cwFill.png" ]]; then
  mkdir -p "$BUNDLE/Contents/Resources"
  cp resources/cwFill.png "$BUNDLE/Contents/Resources/"
fi

echo "built bundle: $BUNDLE  ($(du -h "$DEST/cwFill.ofx" | cut -f1))"
echo "Flame load: set OFX_PLUGIN_PATH=$ROOT/dist  (or run scripts/install.sh)"
