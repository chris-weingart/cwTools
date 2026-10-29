#!/usr/bin/env bash
# Build release artifacts for distribution (beta testers / GitHub Releases).
#
#   macOS (arm64):  dist/release/cwFill-v<ver>-macos-arm64.pkg   (unsigned installer pkg)
#                   dist/release/cwFill-v<ver>-macos-arm64.zip   (pkg + README + INSTALL.txt — the beta hand-off)
#   Linux (x86-64): dist/release/cwFill-v<ver>-linux-x86_64.tar.gz (bundle + install.sh + README + INSTALL.txt)
#
# Run natively per platform (Mac for the mac artifacts, a Linux x86-64
# machine for Linux). Reads the version from Cargo.toml — bump there first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(grep -m1 '^version' crates/ofx-plugin/Cargo.toml | cut -d'"' -f2)"
[[ -n "$VERSION" ]] || { echo "could not read version from Cargo.toml" >&2; exit 1; }

scripts/build.sh

REL="dist/release"
rm -rf "$REL"
mkdir -p "$REL"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp README.md "$STAGE/"

case "$(uname -s)" in
  Darwin)
    NAME="cwFill-v${VERSION}-macos-arm64"

    # Unsigned installer pkg — Installer.app provides the UI, installs as root.
    # Payload comes out of the installer WITHOUT the download-quarantine flag,
    # so no xattr step for testers.
    PKGROOT="$(mktemp -d)"
    mkdir -p "$PKGROOT"
    cp -R dist/cwFill.ofx.bundle "$PKGROOT/"
    pkgbuild --root "$PKGROOT" \
             --identifier dev.weingart.cwFill \
             --version "$VERSION" \
             --install-location /Library/OFX/Plugins \
             "$REL/$NAME.pkg" >/dev/null
    rm -rf "$PKGROOT"

    cat > "$STAGE/INSTALL.txt" <<EOF
cwFill v${VERSION} — macOS (Apple Silicon)

Install (macOS 15 Sequoia and later):
  1. Double-click ${NAME}.pkg. macOS will refuse it ("Apple could not
     verify ... is free of malware"). Click Done — this is expected,
     the beta build is unsigned.
  2. Open System Settings > Privacy & Security and scroll down to the
     Security section. You'll see: "${NAME}.pkg" was blocked to
     protect your Mac. Click "Open Anyway" and confirm.
  3. Follow the installer; the plugin lands in /Library/OFX/Plugins.
  4. Restart your OFX host. Flame: also delete ofxPluginCache.xml
     (/opt/Autodesk/.flame*/status/) so the new plugin is rescanned.

  On macOS 14 and earlier, steps 1-2 are instead: right-click the
  .pkg > Open, then Open again.

Uninstall: sudo rm -rf /Library/OFX/Plugins/cwFill.ofx.bundle

Rather not run an unsigned installer? Inspect it first:
  pkgutil --expand-full ${NAME}.pkg out/
shows the payload is a single bundle for /Library/OFX/Plugins — no
scripts, nothing else. Or skip binaries entirely and build from source
(README, two commands with a Rust toolchain).
EOF

    cp "$REL/$NAME.pkg" "$STAGE/"
    (cd "$STAGE" && zip -qry "$OLDPWD/$REL/$NAME.zip" "$NAME.pkg" README.md INSTALL.txt)

    # Launch (signed) variant — enable once a Developer ID exists:
    #   productsign --sign "Developer ID Installer: <name> (<team>)" \
    #       "$REL/$NAME.pkg" "$REL/$NAME-signed.pkg"
    #   xcrun notarytool submit "$REL/$NAME-signed.pkg" \
    #       --keychain-profile cwfill-notary --wait
    #   xcrun stapler staple "$REL/$NAME-signed.pkg"
    ;;

  Linux)
    NAME="cwFill-v${VERSION}-linux-x86_64"

    cp -R dist/cwFill.ofx.bundle "$STAGE/"
    cp scripts/install.sh "$STAGE/"

    cat > "$STAGE/INSTALL.txt" <<EOF
cwFill v${VERSION} — Linux (x86-64)

Install:
  1. sudo ./install.sh          (copies the bundle to /usr/OFX/Plugins)
  2. Restart your OFX host. Flame: also delete ofxPluginCache.xml
     (/opt/Autodesk/.flame*/status/) so the new plugin is rescanned.

Uninstall: sudo rm -rf /usr/OFX/Plugins/cwFill.ofx.bundle
EOF

    tar -C "$STAGE" -czf "$REL/$NAME.tar.gz" cwFill.ofx.bundle install.sh README.md INSTALL.txt
    ;;

  *)
    echo "unsupported platform: $(uname -s)" >&2
    exit 1
    ;;
esac

echo "release artifacts:"
ls -lh "$REL"
