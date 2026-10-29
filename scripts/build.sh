#!/usr/bin/env bash
# Build cwTools plugins and package dist/<Name>.ofx.bundle (Linux layout:
# Contents/Linux-x86-64/<Name>.ofx). Must be built natively on the target.
#   scripts/build.sh            # all plugins
#   scripts/build.sh cwRefine   # one
#   CWTOOLS_TEST_MODELS=1 scripts/build.sh   # internal: cwRefine with the Synth test models
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.cargo/bin:$PATH"
# no build machine paths in the binaries (panic locations, debug info)
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$ROOT=/src"
# plugin name -> crate (a function, not an associative array: macOS ships bash 3.2)
crate_of() { case "$1" in cwSegment) echo cwsegment ;; cwRefine) echo cwrefine ;; cwMatte) echo cwmatte ;; cwProbe) echo cwprobe ;; esac; }
NAMES=("$@"); [[ ${#NAMES[@]} -eq 0 ]] && NAMES=(cwSegment cwRefine cwMatte)
for NAME in "${NAMES[@]}"; do
  C="$(crate_of "$NAME")"; [[ -n "$C" ]] || { echo "unknown plugin $NAME" >&2; exit 1; }
  VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"  # suite version ([workspace.package])
  FEAT=(); [[ "$C" == cwrefine && "${CWTOOLS_TEST_MODELS:-}" == 1 ]] && FEAT=(--features test-models)
  cargo build --release -p "$C" ${FEAT[@]+"${FEAT[@]}"}
  BUNDLE="dist/$NAME.ofx.bundle"
  rm -rf "$BUNDLE"
  case "$(uname -s)" in
    Linux)  DEST="$BUNDLE/Contents/Linux-x86-64"; LIB="target/release/lib$C.so" ;;
    Darwin) DEST="$BUNDLE/Contents/MacOS";        LIB="target/release/lib$C.dylib" ;;
    *) echo "unsupported OS" >&2; exit 1 ;;
  esac
  mkdir -p "$DEST" "$BUNDLE/Contents/Resources"
  cp "$LIB" "$DEST/$NAME.ofx"
  cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>$NAME.ofx</string>
    <key>CFBundlePackageType</key><string>BNDL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleIdentifier</key><string>dev.weingart.$NAME</string>
</dict>
</plist>
PLIST
  [[ -f "crates/$C/resources/$NAME.png" ]] && cp "crates/$C/resources/$NAME.png" "$BUNDLE/Contents/Resources/"
  if [[ "$(uname -s)" == Darwin ]]; then
    install_name_tool -id "@rpath/$NAME.ofx" "$DEST/$NAME.ofx"  # the linker id is the build path
    codesign --force --sign - "$DEST/$NAME.ofx" 2>/dev/null || true
  fi
  echo "built $BUNDLE ($(du -h "$DEST/$NAME.ofx" | cut -f1)) v$VERSION"
done
