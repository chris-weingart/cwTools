#!/usr/bin/env bash
# Build cwTools install packages from built bundles (scripts/build.sh -> dist/,
# crates/cwfill/scripts/build.sh -> crates/cwfill/dist/). Bundles are OS-specific: build and
# package on the target OS.
#
#   install/make_package.sh                     suite: dist/cwTools-<version>-<os>.tar.gz
#   install/make_package.sh --tool cwRefine     one tool: dist/cwRefine-<version>-<os>.tar.gz
#   install/make_package.sh --tool "cwRefine cwFill"   (or --tool repeated): one package each
#   install/make_package.sh --all-separately    every single-tool package + the suite
#
# Plugin packages hold no models: install.sh downloads them (or uses a local folder or a
# studio share). Model archives, one per set in install/models.tsv (the release tag models-v1):
#   install/make_package.sh --models DIR        cwtools-models-<set>.tar.gz (+ .sha256) for every
#                                               set, files from DIR/<Tool>/ (or DIR/<set>/), each
#                                               file checked against models.tsv; each < 2 GiB
#   install/make_package.sh --models DIR --model-sets "refine-mematte matte-lite"
#   install/make_package.sh --update-models-tsv DIR   refresh sha256 + size in models.tsv from DIR
#
# Options:
#   --with-models DIR         offline package: add the models.tsv files found in DIR/<Tool>/ ->
#                             models/<Tool>/ (cwFill has none). Packages over 2 GiB cannot be
#                             GitHub release assets (cwSegment and cwMatte with all sets, the suite)
#   --with-server-models DIR  suite only: Remote server models DIR/<family>/ -> server_models/
#   --bundles DIR             take every bundle from DIR instead of dist/ + crates/cwfill/dist/
#   --out DIR                 output folder [dist/]
#   --release-names           file names without the version (<Tool>-<os>.tar.gz) for GitHub
#                             release assets, so docs can link releases/latest/download/<file>;
#                             the folder inside and VERSION keep the version
#
# Versions: cwSegment / cwRefine / cwMatte and the suite = [workspace.package] version in
# Cargo.toml (tags vX.Y.Z); cwFill = crates/cwfill/crates/ofx-plugin/Cargo.toml (tags
# cwfill-vX.Y.Z). File names carry the plain version; VERSION / TOOLS inside the package
# add the git describe when the commit is not exactly the release tag.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE_TOOLS="cwSegment cwRefine cwMatte cwFill"
SERVER_FILES="cwinfer_server.py cwtensor.py canary.py netallow.py make_canary.py client_test.py requirements.txt API.md"
TSV="${CWTOOLS_MODELS_TSV:-$REPO/install/models.tsv}"  # override for tests
GIB2=2147483648

TOOLS=""; SUITE=""; ALL_SEP=""; MODELS=""; SMODELS=""; BUNDLES=""; RELNAMES=""; OUT="$REPO/dist"
ARCH_DIR=""; ARCH_SETS=""; UPD_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tool) [ $# -ge 2 ] || { echo "--tool needs a name" >&2; exit 1; }; TOOLS="$TOOLS $2"; shift ;;
    --tool=*) TOOLS="$TOOLS ${1#*=}" ;;
    --suite) SUITE=1 ;;
    --all-separately) ALL_SEP=1 ;;
    --with-models) MODELS="$2"; shift ;;
    --with-server-models) SMODELS="$2"; shift ;;
    --models) ARCH_DIR="$2"; shift ;;
    --model-sets) ARCH_SETS="$2"; shift ;;
    --update-models-tsv) UPD_DIR="$2"; shift ;;
    --bundles) BUNDLES="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --release-names) RELNAMES=1 ;;
    -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 1 ;;
  esac
  shift
done
[ -z "$TOOLS" ] && [ -z "$ALL_SEP" ] && SUITE=1
if [ -n "$ALL_SEP" ]; then TOOLS="$SUITE_TOOLS"; SUITE=1; fi
for t in $TOOLS; do
  case " $SUITE_TOOLS " in *" $t "*) ;; *) echo "unknown tool $t (one of: $SUITE_TOOLS)" >&2; exit 1 ;; esac
done

case "$(uname -s)" in
  Linux) OS=linux; ARCHDIR=Linux-x86-64 ;;
  Darwin) OS=macos; ARCHDIR=MacOS ;;
  *) echo "unsupported OS" >&2; exit 1 ;;
esac
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi

# archives carry no local user / group names (GNU tar on Linux, bsdtar on macOS)
if tar --version 2>/dev/null | grep -q GNU; then TAR_OWNER="--owner=0 --group=0 --numeric-owner"
else TAR_OWNER="--uid 0 --gid 0 --uname root --gname wheel"; fi
tsv_rows() { grep -v '^#' "$TSV" | grep .; }
tsv_sets() { tsv_rows | cut -f1 | awk '!s[$0]++'; }
find_model_file() { # DIR SET FILE -> path of FILE in DIR/<set>/, DIR/<tool>/ or DIR/<family>/
  local tool fam c
  tool="$(tsv_rows | awk -F'\t' -v s="$2" '$1==s {print $2; exit}')"
  fam="$(tsv_rows | awk -F'\t' -v s="$2" '$1==s {print $3; exit}')"
  for c in "$1/$2" "$1/$tool" "$1/$fam"; do [ -f "$c/$3" ] && { echo "$c/$3"; return 0; }; done
  return 1
}
build_archive() { # DIR SET
  local d="$1" x="$2" st f want path got lic L="" total=0 a
  st="$(mktemp -d "${TMPDIR:-/tmp}/cwtools-models.XXXXXX")"; mkdir -p "$st/$x"
  while IFS="$(printf '\t')" read -r _s _t _f _d lic f want _n; do
    [ "$_s" = "$x" ] || continue
    path="$(find_model_file "$d" "$x" "$f")" || { echo "missing for $x: $f (looked in $d/$x, $d/<tool>, $d/<family>)" >&2; rm -rf "$st"; return 1; }
    got="$($SHA < "$path" | awk '{print $1}')"
    [ "$got" = "$want" ] || { echo "sha256 mismatch: $path (got $got, models.tsv says $want; --update-models-tsv after a deliberate change)" >&2; rm -rf "$st"; return 1; }
    ln -s "$path" "$st/$x/$f"; printf '%s  %s\n' "$want" "$f" >> "$st/$x/SHA256SUMS"; L="$lic"
    total=$((total + $(wc -c < "$path")))
  done < <(tsv_rows)
  [ "$total" -lt "$GIB2" ] || { echo "set $x is $total bytes: over GitHub's 2 GiB asset limit, split it in models.tsv" >&2; rm -rf "$st"; return 1; }
  cp "$REPO/licenses/$L" "$st/$x/" || { echo "licence file missing: licenses/$L" >&2; rm -rf "$st"; return 1; }
  printf 'cwTools model set %s. Licence: %s (not cwTools'"'"' Apache-2.0).\nInstall with the cwTools installer: install.sh --models from:<folder with this archive>\n' "$x" "$L" > "$st/$x/README.txt"
  mkdir -p "$OUT"; a="$OUT/cwtools-models-$x.tar.gz"
  COPYFILE_DISABLE=1 tar $TAR_OWNER -h -C "$st" -czf "$a.tmp" "$x" && mv -f "$a.tmp" "$a"
  ( cd "$OUT" && $SHA "cwtools-models-$x.tar.gz" > "cwtools-models-$x.tar.gz.sha256" )
  [ "$(wc -c < "$a")" -lt "$GIB2" ] || { echo "WARNING: $a is over 2 GiB" >&2; }
  echo "built $a ($(du -h "$a" | cut -f1))"
  rm -rf "$st"
}
update_tsv() { # DIR
  local out; out="$(mktemp "${TMPDIR:-/tmp}/models.tsv.XXXXXX")"
  local tab; tab="$(printf '\t')"
  while IFS= read -r line; do
    case "$line" in "#"*|"") echo "$line"; continue ;; esac
    IFS="$tab" read -r x t fam d lic f want n <<< "$line"
    if path="$(find_model_file "$1" "$x" "$f")"; then
      want="$($SHA < "$path" | awk '{print $1}')"; n="$(wc -c < "$path" | tr -d ' ')"
    else echo "not found, row kept: $x $f" >&2; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$x" "$t" "$fam" "$d" "$lic" "$f" "$want" "$n"
  done < "$TSV" > "$out"
  diff "$TSV" "$out" >/dev/null && echo "models.tsv unchanged" || { diff "$TSV" "$out" | sed -n 's/^>/updated:/p'; mv -f "$out" "$TSV"; }
  rm -f "$out"
}

full_version() { # VERSION TAG_PREFIX -> VERSION, or VERSION+<git describe> off the release tag
  local v="$1" id
  # a synced copy without .git carries the id in BUILD_ID
  id="$(cd "$REPO" && git describe --tags --match "$2[0-9]*" --dirty --always 2>/dev/null || head -1 BUILD_ID 2>/dev/null || echo unknown)"
  if [ "$id" = "$2$v" ]; then echo "$v"; else echo "$v+$id"; fi
}
if [ -n "$UPD_DIR" ]; then update_tsv "$(cd "$UPD_DIR" && pwd)"; exit 0; fi
if [ -n "$ARCH_DIR" ]; then
  ARCH_DIR="$(cd "$ARCH_DIR" && pwd)"; OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
  fail=""
  for x in ${ARCH_SETS:-$(tsv_sets)}; do
    tsv_sets | grep -qx "$x" || { echo "unknown set $x" >&2; fail=1; continue; }
    build_archive "$ARCH_DIR" "$x" || fail=1
  done
  [ -z "$fail" ] || { echo "some model archives failed" >&2; exit 1; }
  exit 0
fi
SUITE_VER="$(grep -m1 '^version' "$REPO/Cargo.toml" | cut -d'"' -f2)"
FILL_VER="$(grep -m1 '^version' "$REPO/crates/cwfill/crates/ofx-plugin/Cargo.toml" | cut -d'"' -f2)"
[ -n "$SUITE_VER" ] && [ -n "$FILL_VER" ] || { echo "could not read the versions from Cargo.toml" >&2; exit 1; }
SUITE_FULL="$(full_version "$SUITE_VER" v)"
FILL_FULL="$(full_version "$FILL_VER" cwfill-v)"

tool_ver()  { if [ "$1" = cwFill ]; then echo "$FILL_VER"; else echo "$SUITE_VER"; fi; }
tool_full() { if [ "$1" = cwFill ]; then echo "$FILL_FULL"; else echo "$SUITE_FULL"; fi; }
bundle_src() {
  if [ -n "$BUNDLES" ]; then echo "$BUNDLES/$1.ofx.bundle"
  elif [ "$1" = cwFill ]; then echo "$REPO/crates/cwfill/dist/cwFill.ofx.bundle"
  else echo "$REPO/dist/$1.ofx.bundle"; fi
}

check_bundle() { # NAME
  local b; b="$(bundle_src "$1")"
  [ -d "$b" ] || { echo "not built: $b ($( [ "$1" = cwFill ] && echo crates/cwfill/scripts/build.sh || echo "scripts/build.sh $1"))" >&2; return 1; }
  [ -f "$b/Contents/$ARCHDIR/$1.ofx" ] || { echo "$b has no Contents/$ARCHDIR/$1.ofx (built on another OS?)" >&2; return 1; }
}

verify_models() { # DIR
  [ -f "$1/SHA256SUMS" ] || { echo "no SHA256SUMS in $1" >&2; return 1; }
  if ! ( cd "$1" && $SHA -c SHA256SUMS >/dev/null 2>&1 ); then
    ( cd "$1" && $SHA -c SHA256SUMS 2>&1 | grep -v ': OK$' ) >&2
    echo "model checksum failure in $1" >&2; return 1
  fi
}

copy_tsv_models() { # SRC DST TOOL: only files models.tsv lists for TOOL, sha-checked (+ SHA256SUMS)
  local f want got n=0
  mkdir -p "$2"
  while IFS="$(printf '\t')" read -r _s t _f _d _l f want _n; do
    [ "$t" = "$3" ] && [ -f "$1/$f" ] || continue
    got="$($SHA < "$1/$f" | awk '{print $1}')"
    [ "$got" = "$want" ] || { echo "sha256 mismatch: $1/$f" >&2; return 1; }
    cp "$1/$f" "$2/$f"; printf '%s  %s\n' "$want" "$f" >> "$2/SHA256SUMS"; n=$((n + 1))
  done < <(tsv_rows)
  echo "models: $n files for $3" >&2
}

copy_models_dir() { # SRC DST: only the files SHA256SUMS lists (+ SHA256SUMS)
  mkdir -p "$2"
  cp "$1/SHA256SUMS" "$2/"
  local line name
  while IFS= read -r line; do
    name="$(printf '%s' "$line" | sed -E 's/^[0-9a-fA-F]{64}[[:space:]]+\*?//')"
    [ -n "$name" ] && cp "$1/$name" "$2/$name"
  done < "$1/SHA256SUMS"
}

make_one() { # PACKAGE_NAME PACKAGE_VERSION_FULL "TOOLS" with_server(0/1)
  local name="$1" pver="$2" tools="$3" server="$4" t
  local STAGE; STAGE="$(mktemp -d "${TMPDIR:-/tmp}/cwtools-pkg.XXXXXX")"
  local P="$STAGE/$name"
  mkdir -p "$P/bundles/$OS"
  cp "$REPO/install/install.sh" "$P/install.sh"; chmod 755 "$P/install.sh"
  cp "$TSV" "$P/models.tsv"
  echo "$pver" > "$P/VERSION"
  : > "$P/TOOLS"
  for t in $tools; do
    cp -pR "$(bundle_src "$t")" "$P/bundles/$OS/"
    printf '%s\t%s\n' "$t" "$(tool_full "$t")" >> "$P/TOOLS"
  done
  if [ "$tools" = cwFill ]; then
    cp "$REPO/crates/cwfill/README.md" "$P/README.md"
    cp "$REPO/crates/cwfill/LICENSE" "$P/LICENSE"
  else
    cp "$REPO/README.txt" "$REPO/CHANGELOG.md" "$REPO/LICENSE" "$P/"
    case " $tools " in *" cwFill "*) cp "$REPO/crates/cwfill/README.md" "$P/README-cwFill.md" ;; esac
    # model licences: the shipped weights keep their own licences (THIRD_PARTY_NOTICES.txt)
    cp "$REPO/THIRD_PARTY_NOTICES.txt" "$P/"
    mkdir -p "$P/licenses"; cp "$REPO"/licenses/*.txt "$P/licenses/"
  fi
  # Matchboxes that go with a tool (install.sh puts them in Flame's shared Matchbox folder)
  case " $tools " in *" cwSegment "*)
    mkdir -p "$P/matchbox/cwExtract"
    cp "$REPO"/flame_inference/matchbox/cwExtract/cwExtract.xml "$REPO"/flame_inference/matchbox/cwExtract/cwExtract.*.glsl \
      "$REPO"/flame_inference/matchbox/cwExtract/README.txt "$P/matchbox/cwExtract/" ;;
  esac
  if [ -n "$MODELS" ]; then
    for t in $tools; do
      [ "$t" = cwFill ] && continue
      if [ -d "$MODELS/$t" ]; then copy_tsv_models "$MODELS/$t" "$P/models/$t" "$t"
      else echo "note: no models for $t in $MODELS" >&2; fi
    done
  fi
  if [ "$server" = 1 ]; then
    mkdir -p "$P/server"
    for f in $SERVER_FILES; do cp "$REPO/server/$f" "$P/server/"; done
    if [ -n "$SMODELS" ]; then
      local fam
      for fam in "$SMODELS"/*/; do
        fam="${fam%/}"; [ -f "$fam/SHA256SUMS" ] || continue
        verify_models "$fam" && copy_models_dir "$fam" "$P/server_models/$(basename "$fam")"
      done
    fi
  fi
  mkdir -p "$OUT"
  local file="$name"
  [ -n "${RELNAMES:-}" ] && file="$(printf '%s' "$name" | sed -E 's/-[0-9][^-]*-(linux|macos)$/-\1/')"
  local tgz="$OUT/$file.tar.gz"
  COPYFILE_DISABLE=1 tar $TAR_OWNER -C "$STAGE" -czf "$tgz.tmp" "$name" && mv -f "$tgz.tmp" "$tgz"
  ( cd "$OUT" && $SHA "$file.tar.gz" > "$file.tar.gz.sha256" )
  echo "built $tgz ($(du -h "$tgz" | cut -f1)): $(echo $tools)$([ "$server" = 1 ] && echo " + server")$([ -d "$P/models" ] && echo " + models")"
  [ "$(wc -c < "$tgz")" -lt "$GIB2" ] || echo "WARNING: $tgz is over 2 GiB: too big for a GitHub release asset" >&2
  rm -rf "$STAGE"
}

OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
[ -z "$MODELS" ] || MODELS="$(cd "$MODELS" && pwd)"
[ -z "$SMODELS" ] || SMODELS="$(cd "$SMODELS" && pwd)"
[ -z "$BUNDLES" ] || BUNDLES="$(cd "$BUNDLES" && pwd)"
fail=""
for t in $TOOLS; do check_bundle "$t" || fail=1; done
if [ -n "$SUITE" ]; then for t in $SUITE_TOOLS; do check_bundle "$t" || fail=1; done; fi
[ -z "$fail" ] || { echo "build the missing bundles first" >&2; exit 1; }
[ -z "$SMODELS" ] || [ -n "$SUITE" ] || echo "note: --with-server-models applies to the suite package only" >&2

for t in $TOOLS; do make_one "$t-$(tool_ver "$t")-$OS" "$(tool_full "$t")" "$t" 0; done
if [ -n "$SUITE" ]; then make_one "cwTools-$SUITE_VER-$OS" "$SUITE_FULL" "$SUITE_TOOLS" 1; fi
