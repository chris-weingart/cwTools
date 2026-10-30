#!/usr/bin/env bash
# cwTools installer (Linux + macOS): OFX plugins for Flame, their models and cwtools.conf;
# with --server (Linux only): the cwtools-infer Remote inference server on a GPU machine.
#
#   ./install.sh                 interactive: shows the plan (defaults from the options, an
#                                existing install and what it finds here); Enter installs,
#                                a line number changes that line
#   ./install.sh --yes           accept the plan as shown
#   ./install.sh --check         verify an install (hashes vs manifest, config, servers)
#   ./install.sh --uninstall     remove what the install manifest lists
#   ./install.sh --server ...    Remote inference server (Linux GPU machine)
#   ./install.sh --help          all options
#
# Runs on bash 3.2 (macOS) and bash 4/5 (Linux): no associative arrays, no mapfile,
# no ${var,,}. sudo is used only for the copy steps into directories you cannot write to,
# and only after asking.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_VERSION="$(head -1 "$HERE/VERSION" 2>/dev/null || echo dev)"
MANIFEST_NAME="install-manifest.tsv"
STAMP="$(date +%Y%m%d-%H%M%S)"
TAB="$(printf '\t')"
PLUGIN_NAMES_ALL="cwSegment cwRefine cwMatte cwFill"

case "$(uname -s)" in
  Linux)  OS=linux; DEF_PLUGIN_DIR=/usr/OFX/Plugins;   DEF_ROOT=/opt/cwTools ;;
  Darwin) OS=macos; DEF_PLUGIN_DIR=/Library/OFX/Plugins; DEF_ROOT="/Library/Application Support/cwTools" ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
DEF_SERVER_ROOT=/opt/cwTools/server
DEF_MX_DIR=/opt/Autodesk/shared/matchbox/cwTools
DEF_MODELS_URL=https://github.com/chris-weingart/cwTools/releases/download/models-v1
LOOPBACK_ALLOW="127.0.0.0/8,::1/128"

# ───────────────────────────────────────────────────────────── options ──
YES=""; DRY=""; MODE=install; SERVER=""; PURGE_MODELS=""; FORCE_CONFIG=""; NO_MODELS=""; LIST_MODELS=""
OPT_MODELS=""; OPT_MODEL_SETS=""; OPT_MODELS_URL=""; OPT_ALLOW=""; ALLOW_SET=""
OPT_PLUGIN_DIR=""; OPT_ROOT=""; OPT_MODELS_FROM=""; OPT_PLUGINS=""; OPT_BACKUP_DIR=""
OPT_BUNDLES_FROM=""; OPT_ORT_LIB=""; MX_SET=""; OPT_MX=""
REMOTE_SET=""; OPT_REMOTE=""
OPT_LISTEN=""; OPT_PORT=""; OPT_SERVICE=""; OPT_UNIT=""; OPT_SVC_USER=""; OPT_PYTHON=""
CANARY_CPU=1; NO_TEST=""

usage() {
  cat <<EOF
cwTools installer ($PKG_VERSION)

Plugins (default mode):
  --plugin-dir DIR       OFX plugin folder            [$DEF_PLUGIN_DIR]
                         Another folder also gets launch_flame_cwtools.sh (sets OFX_PLUGIN_PATH)
                         and a copy of cwtools.conf next to the bundles.
  --root DIR             models + cwtools.conf + manifest [$DEF_ROOT]
  --plugins "LIST"       plugins to install            [all bundles in the package]
                         (tools are merged into an existing install: others stay as they are)
  --models MODE          where the models come from (cwFill has none):
                           download      fetch the model archives from --models-url, verify
                                         every file's sha256, install into <root>/models
                           from:DIR      offline: DIR holds cwtools-models-<set>.tar.gz archives,
                                         extracted <set>/ folders, or <Tool>/ folders
                           shared:PATH   studio share: every workstation points cwtools.conf at
                                         PATH (layout PATH/cwRefine, PATH/cwMatte, PATH/cwSegment).
                                         Nothing is copied; the files there are verified. Missing
                                         sets are downloaded into PATH once if you can write there.
                           skip          no models (Remote mode needs none; small sets the
                                         plugin itself reads, such as the cwSegment
                                         tokenizer, are still installed)
                         A bare folder means from:FOLDER.
                         [a previous shared: install keeps its share; else the folder with the
                          most model sets among <package>/models, the package folder, the folder
                          it was unpacked in and ~/Downloads (and their subfolders holding
                          archives); else download when the release answers; else skip]
  --model-sets LIST      "default", "all" or set names (see --list-models) [default]
  --models-url URL       where the archives are         [\$CWTOOLS_MODELS_URL, else
                         $DEF_MODELS_URL]
  --models-from DIR      same as --models from:DIR
  --no-models            same as --models skip
  --list-models          list the model sets (size, default, licence) and exit
  --remote-server URL    cwtools-infer server for Device / Mode = Remote (cwSegment, cwRefine,
                         cwMatte), e.g. http://gpu-box:8201 ["" = none, or the value of an
                         existing cwtools.conf] (the plan shows whether it answers and what
                         kind of server it is)
  --ort-lib auto|PATH    onnxruntime library            [auto = Flame's own]
  --matchbox-dir DIR     where the Matchboxes that come with a plugin go (cwExtract with
                         cwSegment); "" = none        [$DEF_MX_DIR]
  --force-config         with --yes: change values in an existing cwtools.conf (backed up).
                         (Missing keys are always added. A value that differs is changed when
                         you confirm the plan, or when its option is given; --yes alone keeps
                         the existing values.)
  --backup-dir DIR       where replaced bundles go     [<root>/backups] (never the plugin dir)
  --bundles-from DIR     bundle source                 [the package's bundles/$OS/]

Server (--server, Linux only):
  --root DIR             server root                   [$DEF_SERVER_ROOT]
  --models MODE          download | from:DIR | skip, as above (from:DIR also takes the
                         DIR/<family>/ layout: vitmatte, mematte, birefnet, sam3)
                         [from:<package>/server_models if present, else download]
  --model-sets LIST      [default = the server's startup models; "all" = every server set]
  --listen ADDR          listen address                [first non-loopback IPv4 address]
  --port N               port                          [8201]
  --allow CIDRS          clients allowed to connect, comma-separated (CWINFER_ALLOW). The
                         server has no authentication: keep this to your LAN or VPN.
                         [existing value, else 127.0.0.0/8,::1/128 (this machine only)]
  --service system|user|none   systemd system unit (sudo), user unit, or none [system]
  --unit-name NAME       systemd unit name             [cwtools-infer]
  --service-user USER    user the system unit runs as  [you]
  --python PATH          python >= 3.10 for the venv   [newest python3.1x found]
  --no-canary-cpu        skip the slow CPU cross-check when building canaries
  --no-test              skip the client test after start

Common:
  --check                verify the install at --root (hashes, versions, config, servers);
                         with --plugins "LIST" only those tools
  --uninstall            remove what the install manifest at --root lists (asks first);
                         with --plugins "LIST" only those tools
  --purge-models         with --uninstall: also remove the installed models
  --yes, -y              accept all defaults, do not ask (this includes accepting the SAM
                         License for the cwSegment models, licenses/SAM_LICENSE.txt)
  --dry-run              print what would be done, change nothing
  --help, -h

An empty answer keeps the shown default; answer "-" to clear a default (e.g. a server URL).
A wrong answer is explained and asked again. Each install writes a log: install.log in the
cwTools folder (server: the server folder), and a copy in ~/.cache/cwtools/logs.
EOF
}

LOG=""
die()  {
  echo "ERROR: $*" >&2
  [ -n "$LOG" ] && { echo "ERROR: $*" >> "$LOG"; echo "log: $LOG" >&2; }
  exit 1
}
warn() { echo "WARNING: $*"; WARNINGS=$((WARNINGS + 1)); }
info() { echo "$*"; }
WARNINGS=0
CMDLINE="$0$(printf ' %q' "$@")"

while [ $# -gt 0 ]; do
  a="$1"; v=""; has_v=""
  case "$a" in --*=*) v="${a#*=}"; a="${a%%=*}"; has_v=1 ;; esac
  takes_value() {
    case "$a" in --plugin-dir|--root|--models-from|--plugins|--backup-dir|--bundles-from|--ort-lib|\
      --remote-server|--segment-server|--matchbox-dir|--listen|--port|--service|--unit-name|--service-user|--python|\
      --models|--model-sets|--models-url|--allow) return 0 ;; esac
    return 1
  }
  if takes_value; then
    if [ -z "$has_v" ]; then [ $# -ge 2 ] || die "$a needs a value"; v="$2"; shift; fi
  fi
  case "$a" in
    -y|--yes) YES=1 ;;
    --dry-run) DRY=1 ;;
    --check) MODE=check ;;
    --uninstall) MODE=uninstall ;;
    --purge-models) PURGE_MODELS=1 ;;
    --server) SERVER=1 ;;
    --force-config) FORCE_CONFIG=1 ;;
    --no-models) NO_MODELS=1 ;;
    --list-models) LIST_MODELS=1 ;;
    --models) OPT_MODELS="$v" ;;
    --model-sets) OPT_MODEL_SETS="$v" ;;
    --models-url) OPT_MODELS_URL="$v" ;;
    --allow) ALLOW_SET=1; OPT_ALLOW="$v" ;;
    --no-canary-cpu) CANARY_CPU="" ;;
    --no-test) NO_TEST=1 ;;
    --plugin-dir) OPT_PLUGIN_DIR="$v" ;;
    --root) OPT_ROOT="$v" ;;
    --models-from) OPT_MODELS_FROM="$v" ;;
    --plugins) OPT_PLUGINS="$v" ;;
    --backup-dir) OPT_BACKUP_DIR="$v" ;;
    --bundles-from) OPT_BUNDLES_FROM="$v" ;;
    --ort-lib) OPT_ORT_LIB="$v" ;;
    --remote-server) REMOTE_SET=1; OPT_REMOTE="$v" ;;
    --segment-server) die "--segment-server is gone: cwSegment Remote uses the cwtools-infer server (--remote-server)" ;;
    --matchbox-dir) MX_SET=1; OPT_MX="$v" ;;
    --listen) OPT_LISTEN="$v" ;;
    --port) OPT_PORT="$v" ;;
    --service) OPT_SERVICE="$v" ;;
    --unit-name) OPT_UNIT="$v" ;;
    --service-user) OPT_SVC_USER="$v" ;;
    --python) OPT_PYTHON="$v" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

[ -n "$OPT_MODELS_FROM" ] && [ -z "$OPT_MODELS" ] && OPT_MODELS="from:$OPT_MODELS_FROM"
[ -n "$NO_MODELS" ] && OPT_MODELS=skip
MODELS_URL="${OPT_MODELS_URL:-${CWTOOLS_MODELS_URL:-$DEF_MODELS_URL}}"; MODELS_URL="${MODELS_URL%/}"
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/cwtools-install.XXXXXX")" || die "mktemp failed"
trap 'rm -rf "$TMPD"' EXIT

# ───────────────────────────────────────────────────────────── helpers ──
# Prompts read stdin (so answers can be piped in); EOF = accept the default. On a terminal the
# answer can be edited (readline: arrows, backspace). A check function (CHECK ANSWER) prints
# why an answer is wrong and returns 1, and may leave a normalised answer in VALID; a wrong
# answer is asked again (with --yes or at EOF it ends the install).
REPLY_LINE=""; VALID=""
read_line() { # PROMPT -> REPLY_LINE; 1 at EOF
  REPLY_LINE=""
  if [ -t 0 ]; then
    [ -n "$LOG" ] && sleep 0.1 # let the logged output (tee) reach the screen before the prompt
    IFS= read -e -r -p "$1" REPLY_LINE
  else
    printf '%s' "$1"; IFS= read -r REPLY_LINE
  fi
}

ask() { # VAR "question" default [CHECK]
  local __v __c="${4:-}" __eof
  while :; do
    __v="$3"; __eof=""
    if [ -n "$YES" ]; then printf '%s: %s\n' "$2" "${3:-(none)}"
    elif read_line "$2 [${3:-none}]: "; then
      case "$REPLY_LINE" in "") ;; -) __v="" ;; *) __v="$REPLY_LINE" ;; esac
    else echo; __eof=1; fi
    if [ -z "$__c" ] || [ -z "$__v" ]; then break; fi
    VALID="$__v"
    if "$__c" "$__v"; then __v="$VALID"; break; fi
    [ -n "$YES$__eof" ] && die "$2: not valid: $__v"
  done
  [ -n "$LOG" ] && printf '%s: %s\n' "$2" "${__v:-(none)}" >> "$LOG"
  eval "$1=\$__v"
}

yesno() { # "question" y|n -> 0 = yes. --yes takes the default.
  local d="$2"
  if [ -n "$YES" ]; then printf '%s: %s\n' "$1" "$([ "$d" = y ] && echo yes || echo no)"; [ "$d" = y ]; return; fi
  while :; do
    if [ "$d" = y ]; then read_line "$1 [Y/n]: "; else read_line "$1 [y/N]: "; fi || { echo; REPLY_LINE=""; }
    case "$(lc "$REPLY_LINE")" in "") [ "$d" = y ]; return ;; y|yes) return 0 ;; n|no) return 1 ;; esac
  done
}

lc() { printf '%s' "$1" | tr 'A-Z' 'a-z'; }

opt_check() { # OPTION VALUE CHECK -> VALID, or ends the install with the reason
  VALID="$2"; "$3" "$2" || die "$1 $2: see above"
}

# ── checks for answers and options ──
chk_url() { # -> VALID without a trailing /
  case "$1" in http://?*|https://?*) VALID="${1%/}"; return 0 ;; esac
  echo "  a server URL starts with http:// or https:// (e.g. http://gpu-box:8201); - = none"; return 1
}
chk_folder() { VALID="$(abspath "$1")"; [ -n "$VALID" ] || { echo "  a folder, please"; return 1; }; }
chk_port() { case "$1" in ''|*[!0-9]*) echo "  a port is a number (e.g. 8201)"; return 1 ;; esac; }
chk_service() { case "$1" in system|user|none) ;; *) echo "  system, user or none"; return 1 ;; esac; }
chk_user() { id "$1" >/dev/null 2>&1 || { echo "  no such user: $1"; return 1; }; }
chk_allow() {
  VALID="$(printf '%s' "$1" | tr -d ' ')"
  case "$VALID" in *[!0-9A-Fa-f.:/,]*) echo "  comma-separated IPs or CIDRs only (e.g. 10.1.2.0/24,10.0.0.5)"; return 1 ;; esac
}
chk_models_mode() { # download | skip | from:DIR | shared:PATH | a bare folder (= from:) -> VALID
  local d
  case "$1" in
    download|skip) VALID="$1" ;;
    from:*) d="$(abspath "${1#from:}")"
            [ -d "$d" ] || { echo "  folder not found: $d"; return 1; }
            VALID="from:$d" ;;
    shared:*) [ -n "$SERVER" ] && { echo "  shared: is for plugin installs (the server keeps its own copy: from:FOLDER)"; return 1; }
              d="$(abspath "${1#shared:}")"
              [ -n "$d" ] || { echo "  shared: needs a folder"; return 1; }
              [ -d "$d" ] || can_write "$d" || { echo "  shared model folder not found: $d (is the share mounted?)"; return 1; }
              VALID="shared:$d" ;;
    *) d="$(abspath "$1")"
       [ -d "$d" ] || { echo "  download, from:FOLDER, shared:FOLDER or skip (got: $1)"; return 1; }
       VALID="from:$d" ;;
  esac
}
chk_plugins() { # names (any case), numbers from the package list, or all -> VALID (package order)
  local w n i hit out="" bad=""
  for w in $(printf '%s' "$1" | tr ',' ' '); do
    [ "$(lc "$w")" = all ] && { out="$AVAIL"; continue; }
    i=0; hit=""
    for n in $AVAIL; do
      i=$((i + 1))
      { [ "$w" = "$i" ] || [ "$(lc "$w")" = "$(lc "$n")" ]; } && hit="$n"
    done
    if [ -n "$hit" ]; then out="$out $hit"; else bad="$bad $w"; fi
  done
  [ -z "$bad" ] || { echo "  not in this package:$bad (choose from: $(numbered_list "$AVAIL"))"; return 1; }
  VALID=""; for n in $AVAIL; do in_list "$n" "$out" && VALID="$VALID $n"; done
  VALID="${VALID# }"
  [ -n "$VALID" ] || { echo "  choose at least one plugin"; return 1; }
}
chk_sets() { pick_sets "$MT" "$1" && VALID="$MSETS"; } # MT = the tools (or "server")
numbered_list() { local i=0 n; for n in $1; do i=$((i + 1)); [ "$i" -gt 1 ] && printf ', '; printf '%s %s' "$i" "$n"; done; }

confirm() { # like yesno, but --yes means yes
  if [ -n "$YES" ]; then printf '%s: yes (--yes)\n' "$1"; return 0; fi
  yesno "$1" "$2"
}

sam_license_ok() { # the cwSegment models are SAM Materials: ask before installing them
  local lic="$HERE/licenses/SAM_LICENSE.txt"
  echo
  echo "The cwSegment models are Meta's SAM 3, licensed under the SAM License, not Apache-2.0."
  echo "By installing them you agree to its terms (incl. trade controls and prohibited end uses):"
  if [ -f "$lic" ]; then echo "  $lic"; else echo "  https://github.com/facebookresearch/sam3/blob/main/LICENSE"; fi
  confirm "Accept the SAM License and install the cwSegment models?" y
}

abspath() { # expand ~, make absolute, drop a trailing /
  local p="$1"
  case "$p" in "~") p="$HOME" ;; "~/"*) p="$HOME/${p#\~/}" ;; esac
  case "$p" in /*) ;; "") ;; *) p="$PWD/$p" ;; esac
  while [ "${#p}" -gt 1 ] && [ "${p%/}" != "$p" ]; do p="${p%/}"; done
  printf '%s' "$p"
}

if command -v sha256sum >/dev/null 2>&1; then SHA_CMD="sha256sum"; else SHA_CMD="shasum -a 256"; fi
sha_of() { $SHA_CMD < "$1" | awk '{print $1}'; }

xr() { # run a command (or print it with --dry-run). Usage: xr $SUDO_VAR cmd args...
  if [ -n "$DRY" ]; then printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'; return 0; fi
  "$@"
}

nearest_existing() { local d="$1"; while [ ! -e "$d" ]; do d="$(dirname "$d")"; done; printf '%s' "$d"; }

can_write() { # real write test in DIR (or its nearest existing parent)
  local d; d="$(nearest_existing "$1")"
  [ -d "$d" ] || return 1
  local t="$d/.cwtools_write_test.$$"
  if ( : > "$t" ) 2>/dev/null; then rm -f "$t"; return 0; fi
  return 1
}

NEED_SUDO_LIST=""
sudo_for() { # DIR [more paths that must be writable] -> prints "sudo" or ""
  [ "$(id -u)" = 0 ] && return 0
  local p
  for p in "$@"; do
    [ -n "$p" ] || continue
    if [ -d "$p" ] || [ ! -e "$p" ]; then can_write "$p" || { echo sudo; return 0; }
    else [ -w "$p" ] || { echo sudo; return 0; }
    fi
  done
}

ask_sudo() { # "dir list" -> dies if the user refuses
  local dirs="$1"
  [ -n "$dirs" ] || return 0
  echo "Not writable by $(id -un): $dirs"
  [ -n "$DRY" ] && { echo "(dry-run: would use sudo for the copy steps there)"; return 0; }
  command -v sudo >/dev/null 2>&1 || die "sudo not found; run as a user who can write there, or choose other folders"
  confirm "Use sudo for the copy steps into those folders?" y || die "cancelled (choose writable folders with --plugin-dir / --root)"
  # passwordless sudo first: `sudo -v` asks for a password even with NOPASSWD when the
  # user also has a password rule (sudoers default verifypw=all)
  sudo -n true 2>/dev/null || sudo -v || die "sudo failed"
}

put() { # SUDO SRC DST: atomic replace (temp next to the target, then rename), so a running
        # Flame keeps the inode it has mapped and never sees a half-written file.
  local s="$1" tmp; tmp="$(dirname "$3")/.$(basename "$3").tmp.$$"
  xr $s cp "$2" "$tmp" && xr $s mv -f "$tmp" "$3"
}

# Install manifest: one line per item, tab-separated: kind  tool  sha256  path
#   kind: meta (tool = key, path = value) | bundle | bundlefile | model | modelsums | launcher
#         | conf | envfile | doc | version | serverfile | venv | unit
#   tool: the plugin it belongs to, "-" for shared items, "server" for the server
MANIFEST_TMP="$TMPD/manifest.tsv"; : > "$MANIFEST_TMP"
rec()  { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$MANIFEST_TMP"; } # kind tool sha path
meta() { printf 'meta\t%s\t-\t%s\n' "$1" "$2" >> "$MANIFEST_TMP"; }          # key value

manifest_get() { # FILE kind path -> recorded sha (first match)
  [ -f "$1" ] || return 0
  awk -F'\t' -v k="$2" -v p="$3" '$1==k && $4==p {print $3; exit}' "$1"
}
manifest_meta() { # FILE key -> value
  [ -f "$1" ] || return 0
  awk -F'\t' -v k="$2" '$1=="meta" && $2==k {print $4; exit}' "$1"
}
manifest_tools() { # FILE -> installed plugin names (space-separated)
  [ -f "$1" ] || return 0
  awk -F'\t' '$1=="meta" && $2 ~ /^tool:/ {sub(/^tool:/, "", $2); printf "%s ", $2}' "$1" | sed 's/ $//'
}

# Same parsing as the plugins (cwtools-core config.rs): `key = value`, # comments, keys
# case-insensitive, values trimmed and optionally quoted ("..." / '...'), last one wins.
CONF_AWK='function key(l,  i, k) { sub(/^[ \t]+/, "", l); if (l ~ /^#/) return ""; i = index(l, "=")
  if (!i) return ""; k = substr(l, 1, i - 1); gsub(/[ \t]+$/, "", k); return tolower(k) }'
conf_get() { # FILE KEY -> value
  [ -f "$1" ] || return 0
  awk -v want="$2" "$CONF_AWK"'
    { if (key($0) != tolower(want)) next
      v = substr($0, index($0, "=") + 1); gsub(/^[ \t]+|[ \t]+$/, "", v)
      q = substr(v, 1, 1)
      if (length(v) >= 2 && (q == "\"" || q == "\047") && substr(v, length(v)) == q) v = substr(v, 2, length(v) - 2)
      val = v; found = 1 }
    END { if (found) print val }' "$1"
}
conf_has() { [ -f "$1" ] && awk -v want="$2" "$CONF_AWK"' key($0) == tolower(want) { f = 1 } END { exit !f }' "$1"; }

url_check() { # URL -> prints a reachability line for URL/health
  local u="${1%/}" code
  [ -n "$u" ] || return 0
  command -v curl >/dev/null 2>&1 || { echo "  $u: curl not found, not tested"; return 0; }
  code="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$u/health" 2>/dev/null)"
  case "$code" in
    200) echo "  $u/health: reachable (HTTP 200)"; return 0 ;;
    000|"") echo "  $u/health: NOT reachable (no connection within 5 s)"; return 1 ;;
    *) echo "  $u/health: answered HTTP $code"; return 1 ;;
  esac
}

url_ok() { curl -sfL -r 0-0 -m 15 -o /dev/null "$1" 2>/dev/null; } # URL -> 0 when it can be fetched
url_host() { printf '%s' "$1" | sed -e 's#^[A-Za-z]*://##' -e 's#/.*##' -e 's#:[0-9]*$##'; }

# URL -> SK = infer (cwtools-infer, API 2) | infer1 (cwtools-infer 0.3.0: no SAM3 for cwSegment) |
# sam3 (an old SAM3 segmentation server) | forbidden (cwtools-infer, this machine not allowed) |
# other | down | untested (no curl). Cached per URL.
KIND_CACHE=""; SK=""
server_kind() {
  SK="$(printf '%s' "$KIND_CACHE" | awk -F'\t' -v u="$1" '$1==u {print $2; exit}')"
  [ -n "$SK" ] && return 0
  if ! command -v curl >/dev/null 2>&1; then SK=untested
  else
    local body; body="$(curl -s -m 5 "$1/health" 2>/dev/null | head -c 2000)"
    case "$body" in
      "") SK=down ;;
      *'"forbidden"'*) SK=forbidden ;;
      *'"cwtools-infer"'*'"api":1,'*|*'"api":1,'*'"cwtools-infer"'*) SK=infer1 ;;
      *'"cwtools-infer"'*) SK=infer ;;
      *'"features"'*|*'"model_loaded"'*) SK=sam3 ;;
      *) SK=other ;;
    esac
  fi
  KIND_CACHE="$KIND_CACHE$1$TAB$SK
"
}
server_line() { # URL -> "URL  (what answers there)"
  [ -n "$1" ] || { printf 'none'; return 0; }
  server_kind "$1"
  case "$SK" in
    infer) printf '%s  (cwtools-infer, reachable)' "$1" ;;
    infer1) printf '%s  (cwtools-infer 0.3.0, reachable)' "$1" ;;
    sam3) printf '%s  (SAM3 server, reachable)' "$1" ;;
    forbidden) printf '%s  (reachable, but this machine is not on its allow list: CWINFER_ALLOW on the server)' "$1" ;;
    down) printf '%s  (not reachable now; kept anyway)' "$1" ;;
    other) printf '%s  (answers, but is not a cwTools server)' "$1" ;;
    *) printf '%s  (not tested: no curl)' "$1" ;;
  esac
}

detect_flame() {
  local found="" d
  if [ "$OS" = linux ]; then
    for d in /opt/Autodesk/flame_* /opt/Autodesk/flare_* /opt/Autodesk/flameassist_*; do
      [ -d "$d" ] && found="$found  $d
"
    done
  else
    for d in /Applications/Autodesk/Flame*.app /Applications/Autodesk/*/Flame*.app; do
      [ -d "$d" ] && found="$found  $d
"
    done
  fi
  if [ -n "$found" ]; then printf 'Flame installs found:\n%s' "$found"
  else echo "Flame installs found: none (fine: the plugins use the onnxruntime of the Flame that loads them)"
  fi
}

either() { [ -n "$1$2" ] && echo sudo; }
backup_name() { # PATH -> PATH, or PATH.2, PATH.3 ... if taken (backups are never overwritten)
  local p="$1" i=2
  while [ -e "$p" ]; do p="$1.$i"; i=$((i + 1)); done
  printf '%s' "$p"
}
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; } # WORD "LIST"

# what each tool needs
tool_has_models() { in_list "$1" "cwSegment cwRefine cwMatte"; }
tools_need() { # KEY "TOOLS" -> 0 if any of the tools uses that cwtools.conf key
  local t
  for t in $2; do
    case "$1:$t" in
      models_dir:cwSegment|models_dir:cwRefine|models_dir:cwMatte) return 0 ;;
      ort_lib:cwSegment|ort_lib:cwRefine|ort_lib:cwMatte) return 0 ;;
      remote_server:cwSegment|remote_server:cwRefine|remote_server:cwMatte) return 0 ;;
    esac
  done
  return 1
}
tool_version() { # NAME -> version of that tool in this package (TOOLS file), else the package version
  local v; v="$(awk -F'\t' -v n="$1" '$1==n {print $2; exit}' "$HERE/TOOLS" 2>/dev/null)"
  printf '%s' "${v:-$PKG_VERSION}"
}

# ── generic "settings file" handling: env file of the server (whole-file diff + ask) ──
# SUDO NEW TARGET LABEL -> sets CONF_RESULT (written|unchanged|kept)
install_conf_file() {
  local s="$1" new="$2" tgt="$3" label="$4"
  CONF_RESULT=written
  if [ -f "$tgt" ]; then
    if cmp -s "$new" "$tgt"; then CONF_RESULT=unchanged; info "$label unchanged: $tgt"; return 0; fi
    echo "$label exists and differs from what this install would write: $tgt"
    diff -u "$tgt" "$new" | sed "1,2d"
    local replace=""
    if [ -n "$FORCE_CONFIG" ]; then replace=1
    elif [ -z "$YES" ] && yesno "Replace it (the current one is backed up)?" n; then replace=1
    fi
    if [ -z "$replace" ]; then
      CONF_RESULT=kept
      xr $BK_SUDO mkdir -p "$BACKUP_DIR"
      xr $BK_SUDO cp "$new" "$BACKUP_DIR/$(basename "$tgt").proposed-$STAMP"
      info "kept the existing $label; the proposed one is $BACKUP_DIR/$(basename "$tgt").proposed-$STAMP"
      return 0
    fi
    xr $BK_SUDO mkdir -p "$BACKUP_DIR"
    local bn; bn="$(backup_name "$BACKUP_DIR/$(basename "$tgt").$STAMP")"
    xr $BK_SUDO cp "$tgt" "$bn" && info "backup  $bn"
  fi
  xr $s mkdir -p "$(dirname "$tgt")"
  put "$s" "$new" "$tgt" && xr $s chmod 644 "$tgt"
  info "wrote   $tgt"
}

# ── cwtools.conf ──
conf_block() { # KEY VALUE -> comment + assignment for one key
  case "$1" in
    models_dir)
      echo "# Model folders. refine_models / matte_models / segment_models default to"
      echo "# <models_dir>/cwRefine, <models_dir>/cwMatte, <models_dir>/cwSegment."
      echo "models_dir = $2"
      echo "# refine_models = $2/cwRefine"
      echo "# matte_models = $2/cwMatte"
      echo "# segment_models = $2/cwSegment" ;;
    remote_server)
      echo "# Remote inference server (cwtools-infer) for Remote mode in cwSegment, cwRefine and"
      echo "# cwMatte, e.g. http://gpu-box:8201. Empty = none (local GPU only)."
      echo "remote_server = $2" ;;
    ort_lib)
      echo "# onnxruntime library: auto (the one Flame ships) or the path to a libonnxruntime."
      echo "ort_lib = $2" ;;
  esac
}
conf_header() {
  local sysdef
  if [ "$OS" = linux ]; then sysdef=/opt/cwTools/cwtools.conf; else sysdef="/Library/Application Support/cwTools/cwtools.conf"; fi
  echo "# cwTools configuration (written by the cwTools installer; edit freely: the installer"
  echo "# adds missing keys and changes a value only when you choose a new one in its plan)."
  echo "# Format: key = value. Lines starting with # are comments."
  echo "# The plugins read the first of: \$CWTOOLS_CONFIG, cwtools.conf in the folder that holds"
  echo "# the .ofx.bundle folders, $sysdef."
}
conf_want() { # KEY -> value this install wants
  case "$1" in
    models_dir) printf '%s' "$MODELS_DIR" ;;
    remote_server) printf '%s' "$REMOTE" ;;
    ort_lib) printf '%s' "$ORT_LIB" ;;
  esac
}
conf_set_value() { # FILE KEY VALUE -> rewrites the last assignment of KEY in place (to stdout)
  awk -v k="$2" -v v="$3" "$CONF_AWK"'
    NR==FNR { if (key($0) == tolower(k)) last=FNR; next }
    { if (FNR==last) print k " = " v; else print }' "$1" "$1"
}

# SUDO TARGET TOOLS(all installed) OLD_MANIFEST
# New file: header + the keys the tools need. Existing file: missing keys are appended; a
# value that differs from the plan is changed when the plan was confirmed interactively, its
# option was given, or with --force-config (--yes alone keeps it). The diff is shown and the
# old file backed up. models_dir is never changed (except for an explicit shared: install).
update_conf() {
  local s="$1" tgt="$2" tools="$3" oldm="$4" key want have pristine="" changed="" asked="" appended=""
  local new="$TMPD/conf.new"
  CONF_KEYS=""
  for key in models_dir remote_server ort_lib; do
    tools_need "$key" "$tools" && CONF_KEYS="$CONF_KEYS $key"
  done
  [ -n "$CONF_KEYS" ] || { info "cwtools.conf not needed by $tools"; return 0; }
  if [ ! -f "$tgt" ]; then
    pristine=1
    { conf_header; for key in $CONF_KEYS; do echo; conf_block "$key" "$(conf_want "$key")"; done; } > "$new"
  else
    local recsha; recsha="$(manifest_get "$oldm" conf "$tgt")"
    [ -n "$recsha" ] && [ "$recsha" != - ] && [ "$(sha_of "$tgt")" = "$recsha" ] && pristine=1
    cp "$tgt" "$new"
    for key in $CONF_KEYS; do
      want="$(conf_want "$key")"
      if ! conf_has "$new" "$key"; then
        { echo; conf_block "$key" "$want"; } >> "$new"; changed=1; appended=1
      else
        have="$(conf_get "$new" "$key")"
        [ "$have" = "$want" ] && continue
        case "$key" in
          models_dir)
            case "${MMODE:-}" in
              shared:*) # the share was asked for explicitly: point this workstation at it
                conf_set_value "$new" models_dir "$MODELS_DIR" > "$new.t" && mv "$new.t" "$new"
                info "cwtools.conf: models_dir $have -> $MODELS_DIR (shared models)"; changed=1 ;;
              *) info "note: $tgt has models_dir = $have (kept); models were installed to $MODELS_DIR" ;;
            esac ;;
          ort_lib) [ -n "$OPT_ORT_LIB" ] && asked="$asked $key" ;;
          *) asked="$asked $key" ;;
        esac
      fi
    done
    if [ -n "$asked" ]; then
      local ask_new="$TMPD/conf.ask" explicit=""; cp "$new" "$ask_new"
      for key in $asked; do
        conf_set_value "$ask_new" "$key" "$(conf_want "$key")" > "$ask_new.t" && mv "$ask_new.t" "$ask_new"
        case "$key" in remote_server) [ -n "$REMOTE_SET" ] && explicit=1 ;; ort_lib) explicit=1 ;; esac
      done
      if [ -n "$FORCE_CONFIG" ] || [ -z "$YES" ] || [ -n "$explicit" ]; then
        echo "$tgt: changing to the plan's value(s):"
        diff -u "$new" "$ask_new" | sed '1,2d'
        cp "$ask_new" "$new"; changed=1
      else
        echo "$tgt keeps its value(s) (--yes; --force-config or the option itself changes them):"
        diff -u "$new" "$ask_new" | sed '1,2d'
      fi
    fi
    if [ -z "$changed" ]; then
      info "cwtools.conf unchanged: $tgt"
      rec conf - "${recsha:--}" "$tgt"
      return 0
    fi
    [ -n "$appended" ] && { echo "adding missing keys to $tgt:"; diff -u "$tgt" "$new" | sed '1,2d'; }
    xr $BK_SUDO mkdir -p "$BACKUP_DIR"
    local bn; bn="$(backup_name "$BACKUP_DIR/cwtools.conf.$STAMP")"
    xr $BK_SUDO cp "$tgt" "$bn" && info "backup  $bn"
  fi
  xr $s mkdir -p "$(dirname "$tgt")"
  put "$s" "$new" "$tgt" && xr $s chmod 644 "$tgt"
  info "wrote   $tgt"
  # a conf the user edited stays theirs ("-"): uninstall removes only an untouched one
  if [ -n "$pristine" ]; then rec conf - "$(sha_of "$new")" "$tgt"; else rec conf - - "$tgt"; fi
}

# copy one model folder with sha256 verification (re-hash after copy); identical files are
# skipped, missing / corrupt ones are warnings. SUDO SRCDIR DSTDIR TOOL
copy_models() {
  local s="$1" src="$2" dst="$3" tool="$4" sum name f tmp got line n_new=0 n_same=0 n_bad=0
  if [ ! -f "$src/SHA256SUMS" ]; then warn "no SHA256SUMS in $src: models not copied"; return 0; fi
  xr $s mkdir -p "$dst"
  : > "$TMPD/sums.ok"
  while IFS= read -r line <&3; do
    sum="${line%%[ 	]*}"; name="${line#"$sum"}"; name="${name#"${name%%[! 	]*}"}"; name="${name#\*}"
    [ "${#sum}" -eq 64 ] && [ -n "$name" ] || continue
    sum="$(printf '%s' "$sum" | tr 'A-F' 'a-f')"
    f="$src/$name"
    if [ ! -f "$f" ]; then warn "model missing in source: $f"; n_bad=$((n_bad + 1)); continue; fi
    if [ -f "$dst/$name" ] && [ "$(sha_of "$dst/$name")" = "$sum" ]; then
      n_same=$((n_same + 1))
    else
      tmp="$dst/.$name.tmp.$$"
      xr $s cp "$f" "$tmp" || { warn "copy failed: $f"; n_bad=$((n_bad + 1)); continue; }
      if [ -z "$DRY" ]; then
        got="$(sha_of "$tmp")"
        if [ "$got" != "$sum" ]; then
          xr $s rm -f "$tmp"
          warn "sha256 mismatch for $f (got $got, SHA256SUMS says $sum): NOT installed"
          n_bad=$((n_bad + 1)); continue
        fi
      fi
      xr $s mv -f "$tmp" "$dst/$name"
      n_new=$((n_new + 1))
    fi
    printf '%s  %s\n' "$sum" "$name" >> "$TMPD/sums.ok"
    rec "${MODEL_KIND:-model}" "$tool" "$sum" "$dst/$name"
  done 3< "$src/SHA256SUMS"
  # SHA256SUMS in the destination = its old entries not replaced + everything verified now
  if [ -f "$dst/SHA256SUMS" ]; then
    awk 'NR==FNR { n=$0; sub(/^[0-9a-fA-F]+[ \t]+\*?/, "", n); seen[n]=1; next }
         { n=$0; sub(/^[0-9a-fA-F]+[ \t]+\*?/, "", n); if (!(n in seen)) print }' \
      "$TMPD/sums.ok" "$dst/SHA256SUMS" > "$TMPD/sums.merged"
  else
    : > "$TMPD/sums.merged"
  fi
  cat "$TMPD/sums.ok" >> "$TMPD/sums.merged"
  if [ ! -f "$dst/SHA256SUMS" ] || ! cmp -s "$TMPD/sums.merged" "$dst/SHA256SUMS"; then
    put "$s" "$TMPD/sums.merged" "$dst/SHA256SUMS"
  fi
  [ "${MODEL_KIND:-model}" = model ] && rec modelsums "$tool" - "$dst/SHA256SUMS"
  local bad=""; [ "$n_bad" -gt 0 ] && bad=", $n_bad FAILED"
  info "models  $dst: $n_new copied (verified), $n_same identical (skipped)$bad"
}

# ── model sets (models.tsv next to install.sh: one row per file, the sha256 there is what every
#    installed file must match; archives and folders are only containers) ──
MODELS_TSV="$HERE/models.tsv"
mt_rows() { [ -f "$MODELS_TSV" ] || die "models.tsv missing next to install.sh"; grep -v '^#' "$MODELS_TSV" | grep .; }
set_field() { mt_rows | awk -F'\t' -v s="$1" -v c="$2" '$1==s {print $c; exit}'; }   # SET COLUMN
set_bytes() { mt_rows | awk -F'\t' -v s="$1" '$1==s {n += $8} END {printf "%.0f", n}'; }
set_files() { mt_rows | awk -F'\t' -v s="$1" '$1==s {print $6}'; }
set_exists() { [ -n "$(set_field "$1" 1)" ]; }
human() { awk -v b="$1" 'BEGIN { if (b >= 1e9) printf "%.1f GB", b / 1e9; else printf "%.0f MB", b / 1e6 }'; }
# sets of TOOLS (plugin) or of the server (TOOLS = server); WHICH = default | all
sets_for() { # "TOOLS" WHICH
  mt_rows | awk -F'\t' -v tools=" $1 " -v which="$2" -v os="$OS" '
    { srv = (tools == " server "); ok = srv ? ($3 != "-") : index(tools, " " $2 " ")
      if (!ok || seen[$1]++) next
      # l / m = plugin default on Linux / macOS only (no CUDA on macOS: CPU models)
      d = srv ? index($4, "s") : (index($4, "p") || (os == "macos" && index($4, "m")) || (os == "linux" && index($4, "l")))
      if (which == "default" && !d) next
      printf "%s ", $1 }' | sed 's/ $//'
}
# sets every install of TOOLS gets whatever the models mode, skip included (default flag r)
always_sets() { # "TOOLS"
  mt_rows | awk -F'\t' -v tools=" $1 " 'index(tools, " " $2 " ") && index($4, "r") && !seen[$1]++ {printf "%s ", $1}' | sed 's/ $//'
}
sets_total() { local t=0 x; for x in $1; do t=$((t + $(set_bytes "$x"))); done; echo "$t"; }
list_models() { # "TOOLS" -> table
  echo "Model sets (archives cwtools-models-<set>.tar.gz; * = installed by default):"
  local x
  for x in $(mt_rows | cut -f1 | awk '!s[$0]++'); do
    [ -n "$1" ] && ! in_list "$(set_field "$x" 2)" "$1" && [ "$1" != server ] && continue
    [ "$1" = server ] && [ "$(set_field "$x" 3)" = - ] && continue
    printf '  %-20s %-9s %8s  %-17s (licence %s)\n' "$x" "$(set_field "$x" 2)" "$(human "$(set_bytes "$x")")" \
      "$(default_marks "$(set_field "$x" 4)")" "$(set_field "$x" 5 | sed 's/_LICENSE.txt//')"
  done
}
default_marks() { # p|s|ps|- -> "* plugin * server"
  local m=""
  case "$1" in *p*) m="* plugin" ;; esac
  case "$1" in *l*) m="${m:+$m }* Linux" ;; esac
  case "$1" in *m*) m="${m:+$m }* macOS" ;; esac
  case "$1" in *s*) m="${m:+$m }* server" ;; esac
  case "$1" in *r*) m="${m:+$m }* always" ;; esac
  printf '%s' "$m"
}
pick_sets() { # "TOOLS" ANSWER -> MSETS; 1 (with the reason printed) for an unknown set
  local x out=""
  case "$(lc "$2")" in
    ""|default) out="$(sets_for "$1" default)" ;;
    all) out="$(sets_for "$1" all)" ;;
    *) for x in $(lc "$2" | tr ',' ' '); do
         set_exists "$x" || { echo "  unknown model set: $x (see the list / --list-models)"; return 1; }
         in_list "$x" "$(sets_for "$1" all)" || { echo "  model set $x is not for $1"; return 1; }
         in_list "$x" "$out" || out="$out $x"
       done ;;
  esac
  MSETS="${out# }"
}
file_size() { wc -c < "$1" | tr -d ' '; }
free_bytes() { df -Pk "$(nearest_existing "$1")" 2>/dev/null | awk 'NR==2 {printf "%.0f", $4 * 1024}'; }
# "SETS" -> bytes the install adds under MODELS_DIR (files already there at their size count 0),
# plus the largest set again for unpacking
sets_need_bytes() {
  local x t f b sb tot=0 big=0
  for x in $1; do
    t="$(set_field "$x" 2)"; sb=0
    while IFS="$TAB" read -r f b; do
      [ -f "$MODELS_DIR/$t/$f" ] && [ "$(file_size "$MODELS_DIR/$t/$f")" = "$b" ] && continue
      sb=$((sb + b))
    done < <(mt_rows | awk -F'\t' -v s="$x" '$1==s {print $6 "\t" $8}')
    tot=$((tot + sb)); [ "$sb" -gt "$big" ] && big="$sb"
  done
  echo $((tot > 0 ? tot + big : 0))
}

MODELS_CACHE="${CWTOOLS_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/cwtools}/models"
fetch() { # URL OUT: resumable, retried
  curl -fL -# --retry 5 --retry-delay 3 --connect-timeout 20 -C - -o "$2" "$1"
}
# SET -> downloads + extracts the archive, prints the extracted folder (with SHA256SUMS from
# models.tsv). The archive is checked against its .sha256 when the release has one.
download_set() {
  local x="$1" a="cwtools-models-$1.tar.gz" want got try
  command -v curl >/dev/null 2>&1 || die "curl not found (needed to download models; or use --models from:DIR)"
  mkdir -p "$MODELS_CACHE" || die "cannot create $MODELS_CACHE (set CWTOOLS_CACHE)"
  want=""
  if curl -fsSL --retry 3 -o "$MODELS_CACHE/$a.sha256" "$MODELS_URL/$a.sha256" 2>/dev/null; then
    want="$(awk '{print tolower($1); exit}' "$MODELS_CACHE/$a.sha256")"
  fi
  for try in 1 2; do
    if [ ! -f "$MODELS_CACHE/$a" ]; then
      echo "download $MODELS_URL/$a ($(human "$(set_bytes "$x")"))" >&2
      # an interrupted download resumes (-C -). When resuming fails (complete .part = 416, or
      # no byte ranges) the archive sha256 decides; without one, start over once.
      if ! fetch "$MODELS_URL/$a" "$MODELS_CACHE/$a.part" >&2; then
        if [ -s "$MODELS_CACHE/$a.part" ] && [ -z "$want" ]; then
          rm -f "$MODELS_CACHE/$a.part"
          fetch "$MODELS_URL/$a" "$MODELS_CACHE/$a.part" >&2 || { echo "download failed: $MODELS_URL/$a" >&2; return 1; }
        elif [ ! -s "$MODELS_CACHE/$a.part" ]; then
          echo "download failed: $MODELS_URL/$a (run again to resume)" >&2; return 1
        fi
      fi
      mv -f "$MODELS_CACHE/$a.part" "$MODELS_CACHE/$a"
    fi
    got="$(sha_of "$MODELS_CACHE/$a")"
    if [ -z "$want" ] || [ "$got" = "$want" ]; then break; fi
    rm -f "$MODELS_CACHE/$a"
    [ "$try" = 2 ] && { echo "archive $a fails its sha256 again (got $got, release says $want): not installed" >&2; return 1; }
    echo "archive sha256 mismatch for $a (got $got, release says $want): downloading again" >&2
  done
  rm -rf "$MODELS_CACHE/stage/$x"; mkdir -p "$MODELS_CACHE/stage"
  tar -xzf "$MODELS_CACHE/$a" -C "$MODELS_CACHE/stage" || { echo "cannot extract $a (removed; run again)" >&2; rm -f "$MODELS_CACHE/$a"; return 1; }
  [ -d "$MODELS_CACHE/stage/$x" ] || { echo "$a has no $x/ folder" >&2; return 1; }
  set_sums "$x" > "$MODELS_CACHE/stage/$x/SHA256SUMS"
  printf '%s' "$MODELS_CACHE/stage/$x"
}
set_sums() { mt_rows | awk -F'\t' -v s="$1" '$1==s {print $7 "  " $6}'; }
# DIR SET -> where DIR holds the set: "archive" (cwtools-models-<set>.tar.gz), or the folder
# with all its files: an extracted <set>/, a <Tool>/ (package models/) or a <family>/ (server)
set_source() {
  local d="$1" x="$2" f c
  [ -f "$d/cwtools-models-$x.tar.gz" ] && { printf archive; return 0; }
  for c in "$d/$x" "$d/$(set_field "$x" 2)" "$d/$(set_field "$x" 3)"; do
    [ -d "$c" ] || continue
    for f in $(set_files "$x"); do [ -f "$c/$f" ] || continue 2; done
    printf '%s' "$c"; return 0
  done
  return 1
}
sets_present() { local x o=""; for x in $2; do set_source "$1" "$x" >/dev/null && o="$o $x"; done; printf '%s' "${o# }"; } # DIR "SETS"
sets_missing() { local x o=""; for x in $2; do set_source "$1" "$x" >/dev/null || o="$o $x"; done; printf '%s' "${o# }"; } # DIR "SETS"
# DIR SET -> prints a folder holding the set's files + SHA256SUMS
local_set() {
  local d="$1" x="$2" f c st
  c="$(set_source "$d" "$x")" || return 1
  if [ "$c" = archive ]; then
    st="$TMPD/stage"; rm -rf "$st/$x"; mkdir -p "$st"
    tar -xzf "$d/cwtools-models-$x.tar.gz" -C "$st" && [ -d "$st/$x" ] || { echo "cannot extract $d/cwtools-models-$x.tar.gz" >&2; return 1; }
    set_sums "$x" > "$st/$x/SHA256SUMS"; printf '%s' "$st/$x"; return 0
  fi
  st="$TMPD/link/$x"; rm -rf "$st"; mkdir -p "$st"
  for f in $(set_files "$x"); do ln -s "$c/$f" "$st/$f"; done
  set_sums "$x" > "$st/SHA256SUMS"; printf '%s' "$st"
}
# folders that may hold model archives or model folders, one per line
model_dirs() {
  local b f
  {
    printf '%s\n' "$HERE/models" "$HERE"
    for b in "$(dirname "$HERE")" "$HOME/Downloads"; do
      [ -d "$b" ] || continue
      printf '%s\n' "$b"
      for f in "$b"/*/cwtools-models-*.tar.gz; do [ -f "$f" ] && dirname "$f"; done
    done
  } | awk '!s[$0]++'
}
# default model source -> DM: an earlier shared: install keeps its share; else the folder
# with the most of the tools' sets; else download when the release answers; else skip
REL_OK=""
release_ok() {
  if [ -z "$REL_OK" ]; then
    local x; x="$(sets_for "${MT:-cwRefine}" all | awk '{print $1}')"
    REL_OK=n; command -v curl >/dev/null 2>&1 && url_ok "$MODELS_URL/cwtools-models-$x.tar.gz" && REL_OK=y
  fi
  [ "$REL_OK" = y ]
}
default_models_mode() {
  case "$OLD_MM" in shared:*) DM="$OLD_MM"; return 0 ;; esac
  local d n best="" bn=0 all; all="$(sets_for "$MT" all)"
  while IFS= read -r d; do
    [ -d "$d" ] || continue
    n="$(sets_present "$d" "$all" | wc -w | tr -d ' ')"
    [ "$n" -gt "$bn" ] && { best="$d"; bn="$n"; }
  done < <(model_dirs)
  if [ -n "$best" ]; then DM="from:$best"
  elif release_ok; then DM=download
  else DM=skip; fi
}
# SET DIR -> 0 when every file of the set is in DIR with the right sha256 (prints problems)
verify_set_at() {
  local x="$1" d="$2" f want bad=0
  for f in $(set_files "$x"); do
    want="$(mt_rows | awk -F'\t' -v s="$x" -v f="$f" '$1==s && $6==f {print $7; exit}')"
    if [ ! -f "$d/$f" ]; then echo "  missing  $d/$f"; bad=1
    elif [ "$(sha_of "$d/$f")" != "$want" ]; then echo "  CHANGED  $d/$f (sha256 differs from models.tsv)"; bad=1; fi
  done
  return "$bad"
}
# SUDO MODE SET DST TOOL: install one set into DST with the chosen source
install_set() {
  local s="$1" mode="$2" x="$3" dst="$4" tool="$5" src=""
  case "$mode" in
    download) [ -n "$DRY" ] && { echo "  [dry-run] download + verify $x -> $dst"; return 0; }
              [ -f "$MODELS_CACHE/cwtools-models-$x.tar.gz.part" ] || url_ok "$MODELS_URL/cwtools-models-$x.tar.gz" \
                || { warn "model set $x not installed: no archive at $MODELS_URL/cwtools-models-$x.tar.gz"; return 0; }
              src="$(download_set "$x")" || { warn "model set $x not installed (download failed; later: install.sh --models download --model-sets $x)"; return 0; } ;;
    from:*)   src="$(local_set "${mode#from:}" "$x")" || { warn "model set $x not found in ${mode#from:}"; return 0; } ;;
  esac
  copy_models "$s" "$src" "$dst" "$tool"
  case "$src" in
    "$MODELS_CACHE"/stage/*) [ -z "$DRY" ] && rm -rf "$src" "$MODELS_CACHE/cwtools-models-$x.tar.gz" "$MODELS_CACHE/cwtools-models-$x.tar.gz.sha256" ;;
    "$TMPD"/stage/*) rm -rf "$src" ;; # unpacked archive: free the space before the next set
  esac
}
# SET: an always set (always_sets) is installed in every models mode. Already verified in place:
# kept; from:FOLDER with the set: from there; else from the release (shared: verifies the share).
install_always() {
  local x="$1" n f; n="$(set_field "$x" 2)"
  case "$MMODE" in shared:*) shared_set "$x" "$MODELS_DIR/$n" "$n"; return 0 ;; esac
  if [ -z "$DRY" ] && verify_set_at "$x" "$MODELS_DIR/$n" >/dev/null; then
    for f in $(set_files "$x"); do
      rec model "$n" "$(mt_rows | awk -F'\t' -v s="$x" -v f="$f" '$1==s && $6==f {print $7; exit}')" "$MODELS_DIR/$n/$f"
    done
    info "models  $MODELS_DIR/$n: $x verified (kept)"; return 0
  fi
  case "$MMODE" in
    from:*) if set_source "${MMODE#from:}" "$x" >/dev/null; then
              install_set "$R_SUDO" "$MMODE" "$x" "$MODELS_DIR/$n" "$n"; return 0
            fi ;;
  esac
  install_set "$R_SUDO" download "$x" "$MODELS_DIR/$n" "$n"
}
# SUDO DST TOOL: SHA256SUMS in DST lists every model of TOOL in models.tsv, installed or not.
# Device = Remote names a model by its sha256 (the plugins read SHA256SUMS before hashing the
# file), so Remote works for models that are only on the server. Other lines stay.
known_sums() {
  local s="$1" dst="$2" tool="$3" x
  : > "$TMPD/known"
  for x in $(sets_for "$tool" all); do set_sums "$x" >> "$TMPD/known"; done
  [ -s "$TMPD/known" ] || return 0
  if [ -f "$dst/SHA256SUMS" ]; then
    awk 'NR==FNR { n=$0; sub(/^[0-9a-fA-F]+[ \t]+\*?/, "", n); seen[n]=1; next }
         { n=$0; sub(/^[0-9a-fA-F]+[ \t]+\*?/, "", n); if (!(n in seen)) print }' \
      "$TMPD/known" "$dst/SHA256SUMS" > "$TMPD/known.merged"
  else
    : > "$TMPD/known.merged"
  fi
  cat "$TMPD/known" >> "$TMPD/known.merged"
  if [ ! -f "$dst/SHA256SUMS" ] || ! cmp -s "$TMPD/known.merged" "$dst/SHA256SUMS"; then
    xr $s mkdir -p "$dst"
    put "$s" "$TMPD/known.merged" "$dst/SHA256SUMS" && xr $s chmod 644 "$dst/SHA256SUMS"
  fi
  rec modelsums "$tool" - "$dst/SHA256SUMS"
}
# shared:PATH: verify the set in PATH/<Tool>; record the files (kind extmodel: checked by
# --check, never removed by --uninstall); offer to download what is missing once
shared_set() { # SET DST TOOL
  local x="$1" dst="$2" tool="$3" f
  if ! verify_set_at "$x" "$dst"; then
    if can_write "$dst" && confirm "Model set $x is missing or incomplete in the shared folder $dst. Download it there now?" y; then
      install_set "" download "$x" "$dst" "$tool"   # records the files itself (MODEL_KIND)
      return 0
    else
      warn "model set $x is not complete in $dst: put it there from a workstation that can write the share (install.sh --models shared:$(dirname "$dst") --model-sets $x)"
    fi
  else
    info "models  $dst: $x verified (shared, nothing copied)"
  fi
  for f in $(set_files "$x"); do
    [ -f "$dst/$f" ] && rec extmodel "$tool" "$(mt_rows | awk -F'\t' -v s="$x" -v f="$f" '$1==s && $6==f {print $7; exit}')" "$dst/$f"
  done
  return 0
}

# SUDO TARGET OLD_MANIFEST "TOOLS NOT TOUCHED NOW": merge the previous manifest:
# entries of other tools are kept, model entries of earlier installs are carried forward
finish_manifest() {
  local s="$1" tgt="$2" old="$3" keep="$4" k t h p
  if [ -f "$old" ]; then
    # an old line is superseded by a new one of the same kind + path (meta: same key)
    awk -F'\t' 'NR==FNR { if ($1=="meta") have["meta" FS $2]=1; else have[$1 FS $4]=1; next }
                 !((($1=="meta") ? ("meta" FS $2) : ($1 FS $4)) in have)' "$MANIFEST_TMP" "$old" > "$TMPD/carry"
    while IFS="$TAB" read -r k t h p <&3; do
      if [ "$k" = meta ]; then
        case "$t" in tool:*) in_list "${t#tool:}" "$keep" && printf '%s\t%s\t%s\t%s\n' "$k" "$t" "$h" "$p" >> "$MANIFEST_TMP" ;; esac
        continue
      fi
      if in_list "$t" "$keep"; then printf '%s\t%s\t%s\t%s\n' "$k" "$t" "$h" "$p" >> "$MANIFEST_TMP"
      elif { [ "$k" = model ] || [ "$k" = modelsums ]; } && [ -f "$p" ]; then
        printf '%s\t%s\t%s\t%s\n' "$k" "$t" "$h" "$p" >> "$MANIFEST_TMP"
      fi
    done 3< "$TMPD/carry"
  fi
  awk '!seen[$0]++' "$MANIFEST_TMP" > "$MANIFEST_TMP.u" && mv "$MANIFEST_TMP.u" "$MANIFEST_TMP"
  put "$s" "$MANIFEST_TMP" "$tgt" && xr $s chmod 644 "$tgt"
}

write_version_file() { # SUDO MANIFEST(final, in TMP) TARGET
  awk -F'\t' '$1=="meta" && $2 ~ /^tool:/ {sub(/^tool:/, "", $2); print $2 " " $4}' "$2" > "$TMPD/VERSION"
  printf 'installed %s\n' "$(date '+%Y-%m-%d %H:%M')" >> "$TMPD/VERSION"
  put "$1" "$TMPD/VERSION" "$3"
}

check_manifest_files() { # MANIFEST [TOOLS] -> prints problems, sets CHK_BAD / CHK_OK
  CHK_BAD=0; CHK_OK=0
  local k t h p
  while IFS="$TAB" read -r k t h p <&3; do
    [ -n "$2" ] && [ "$t" != - ] && ! in_list "$t" "$2" && continue
    case "$k" in
      bundlefile|model|extmodel|launcher|serverfile|doc|unit|mxfile)
        if [ ! -f "$p" ]; then echo "  MISSING  $p"; CHK_BAD=$((CHK_BAD + 1))
        elif [ "$(sha_of "$p")" != "$h" ]; then echo "  CHANGED  $p"; CHK_BAD=$((CHK_BAD + 1))
        else CHK_OK=$((CHK_OK + 1)); fi ;;
      conf|envfile)
        if [ ! -f "$p" ]; then echo "  missing  $p (config)"
        elif [ "$h" != - ] && [ "$(sha_of "$p")" != "$h" ]; then echo "  edited   $p (config, edited since install: fine)"
        fi ;;
    esac
  done 3< "$1"
}

# ── install log: the plan + everything printed from the install step on (stdout through tee)
# goes to ~/.cache/cwtools/logs/install-<time>.log, appended to <folder>/install.log at the end
start_log() { # [function printing the plan]
  local d="${CWTOOLS_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/cwtools}/logs"
  mkdir -p "$d" 2>/dev/null || return 0
  LOG="$d/install-$STAMP.log"
  { echo "cwTools install.sh $PKG_VERSION  $(date '+%Y-%m-%d %H:%M:%S')  $(id -un)@$(hostname 2>/dev/null)  $OS"
    echo "command: $CMDLINE"
    [ -n "$1" ] && "$1"
    echo; } > "$LOG" 2>/dev/null || { LOG=""; return 0; }
  exec 5>&1
  exec > >(tee -a "$LOG")
}
finish_log() { # SUDO FOLDER
  [ -n "$LOG" ] || return 0
  if [ -n "$DRY" ]; then echo "  log      $LOG"; else echo "  log      $2/install.log"; fi
  exec 1>&5 5>&-
  sleep 0.3 # tee writes the last lines
  { [ -f "$2/install.log" ] && cat "$2/install.log"; cat "$LOG"; echo; } > "$TMPD/install.log"
  if [ -z "$DRY" ]; then put "$1" "$TMPD/install.log" "$2/install.log" && $1 chmod 644 "$2/install.log"; fi
}

# ═══════════════════════════════════════════════════════ plugin install ══
problem() { PROBLEMS="$PROBLEMS$1
"; }

# The plan: every line starts from a default (options, the existing install, what is found on
# this machine) and is recomputed after each change. Sets the plugins_install locals.
plan_derive() {
  local n c others=""
  PROBLEMS=""
  OLD_MANIFEST="$ROOT/$MANIFEST_NAME"
  OLD_PD="$(manifest_meta "$OLD_MANIFEST" plugin_dir)"
  OLD_TOOLS="$(manifest_tools "$OLD_MANIFEST")"
  OLD_MM="$(manifest_meta "$OLD_MANIFEST" models_mode)"
  # an existing install's plugin folder is the default: tools are merged into it
  [ -n "$CH_PD" ] || PLUGIN_DIR="${OLD_PD:-$DEF_PLUGIN_DIR}"
  if [ -n "$OLD_PD" ] && [ "$OLD_PD" != "$PLUGIN_DIR" ]; then
    for n in $OLD_TOOLS; do in_list "$n" "$PLUGINS" || others="$others $n"; done
    [ -z "$others" ] || problem "line 3: the install in $ROOT has$others in $OLD_PD: use that folder, or another cwTools folder (line 2)"
  fi
  BACKUP_DIR="$(abspath "${OPT_BACKUP_DIR:-$(manifest_meta "$OLD_MANIFEST" backup_dir)}")"
  BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
  case "$BACKUP_DIR/" in "$PLUGIN_DIR"/*) problem "the backup folder $BACKUP_DIR is inside the plugin folder: Flame would load the backups (--backup-dir)" ;; esac
  case "$ROOT/" in "$PLUGIN_DIR"/*) problem "line 2: the cwTools folder is inside the plugin folder (backups would end up there)" ;; esac
  MODELS_DIR="$ROOT/models"
  STD_PLUGIN=""; STD_ROOT=""
  [ "$PLUGIN_DIR" = "$DEF_PLUGIN_DIR" ] && STD_PLUGIN=1
  [ "$ROOT" = "$DEF_ROOT" ] && STD_ROOT=1
  ALL_TOOLS="$PLUGINS"; for n in $OLD_TOOLS; do in_list "$n" "$ALL_TOOLS" || ALL_TOOLS="$ALL_TOOLS $n"; done
  KEEP_TOOLS=""; for n in $OLD_TOOLS; do in_list "$n" "$PLUGINS" || KEEP_TOOLS="$KEEP_TOOLS $n"; done
  KEEP_TOOLS="${KEEP_TOOLS# }"

  # models (only for tools that have models): source, sets, disk space
  MT=""; for n in $PLUGINS; do tool_has_models "$n" && MT="$MT $n"; done
  MT="${MT# }"
  MODEL_KIND=model; MSETS=""; NEED_B=0; FREE_B=""
  if [ -z "$MT" ]; then MMODE=skip
  else
    [ -n "$CH_MM" ] || { default_models_mode; MMODE="$DM"; }
    case "$MMODE" in shared:*) MODELS_DIR="${MMODE#shared:}"; MODEL_KIND=extmodel ;; esac
    if [ "$MMODE" != skip ]; then
      if [ -n "$CH_SETS" ]; then
        pick_sets "$MT" "$MSETS_ANS" > "$TMPD/msg" || problem "line 5:$(cat "$TMPD/msg")"
      else
        MSETS="$(sets_for "$MT" default)"
        case "$MMODE" in from:*) MSETS="$(sets_present "${MMODE#from:}" "$MSETS")" ;; esac
      fi
      case "$MMODE" in
        from:*) c="$(sets_missing "${MMODE#from:}" "$MSETS")"
                [ -z "$c" ] || problem "line 5: not in ${MMODE#from:}: $c" ;;
        download) release_ok || problem "line 4: no model archives at $MODELS_URL (not published yet, or offline): use from:FOLDER, or skip (Device = Remote only)" ;;
      esac
      case "$MMODE" in download|from:*)
        NEED_B="$(sets_need_bytes "$MSETS")"; FREE_B="$(free_bytes "$MODELS_DIR")"
        if [ -n "$FREE_B" ] && [ "$FREE_B" -lt "$NEED_B" ]; then
          problem "line 5: needs $(human "$NEED_B") free on the disk of $MODELS_DIR, $(human "$FREE_B") free: fewer sets (line 5), or skip (line 4)"
        fi ;;
      esac
    fi
  fi

  # Matchboxes of the chosen plugins (package matchbox/<name>/)
  MXS=""
  if in_list cwSegment "$PLUGINS" && [ -d "$HERE/matchbox/cwExtract" ]; then MXS=cwExtract; fi
  [ -n "$CH_MX" ] || MX_DIR="$(manifest_meta "$OLD_MANIFEST" matchbox_dir)"
  [ -n "$CH_MX" ] || MX_DIR="${MX_DIR:-$DEF_MX_DIR}"
  case "$MX_DIR/" in "$PLUGIN_DIR"/*) problem "line 7: the Matchbox folder is inside the plugin folder" ;; esac
  # Flame's Matchbox browser opens in <presets>/<version>/matchbox/shaders: cwExtract goes there too
  MX_PRESETS=""
  if [ -n "$MXS" ] && [ -n "$MX_DIR" ]; then
    for c in /opt/Autodesk/presets/*/matchbox/shaders; do [ -d "$c" ] && MX_PRESETS="$MX_PRESETS$c
"; done
  fi

  # servers: defaults from an existing cwtools.conf
  OLD_CONF=""
  for c in "$PLUGIN_DIR/cwtools.conf" "$ROOT/cwtools.conf"; do [ -f "$c" ] && { OLD_CONF="$c"; break; }; done
  [ -n "$CH_REMOTE" ] || REMOTE="$(conf_get "$OLD_CONF" remote_server)"
  ORT_LIB="${OPT_ORT_LIB:-$(conf_get "$OLD_CONF" ort_lib)}"; ORT_LIB="${ORT_LIB:-auto}"
  REMOTE_NOTE=""
  if tools_need remote_server "$PLUGINS" && [ -n "$REMOTE" ]; then
    server_kind "$REMOTE"
    [ "$SK" = sam3 ] && problem "line 6: $REMOTE is a SAM3 server; the plugins need cwtools-infer (port 8201 by default)"
    [ "$SK" = infer1 ] && in_list cwSegment "$PLUGINS" && REMOTE_NOTE="cwSegment Remote needs cwtools-infer 0.3.1 or newer there: update the server (install.sh --server)"
  fi
  return 0
}

plan_show() {
  local n x has noloc="" v
  echo
  echo "Plan:"
  echo "  1  plugins         $(for n in $PLUGINS; do printf '%s %s  ' "$n" "$(tool_version "$n")"; done)"
  [ -n "$KEEP_TOOLS" ] && echo "                     already installed, kept as they are: $KEEP_TOOLS"
  echo "  2  cwTools folder  $ROOT$([ -n "$OLD_TOOLS" ] && printf '  (existing install: %s)' "$OLD_TOOLS")"
  echo "  3  OFX folder      $PLUGIN_DIR$([ -z "$STD_PLUGIN" ] && printf '  (not Flame'"'"'s standard folder: a launcher goes there)')"
  if [ -n "$MT" ]; then
    case "$MMODE" in
      download) v="download from the release" ;;
      from:*) v="copy from ${MMODE#from:}" ;;
      shared:*) v="studio share ${MMODE#shared:} (nothing copied; cwtools.conf points there)" ;;
      *) v="none (Device = Remote only)$(a="$(always_sets "$MT")"; [ -n "$a" ] && printf '; %s (%s) still installed' "$a" "$(human "$(sets_total "$a")")")" ;;
    esac
    echo "  4  models          $v"
    if [ "$MMODE" != skip ]; then
      echo "  5  model sets      ${MSETS:-none}$([ -n "$MSETS" ] && printf '  (%s' "$(human "$(sets_total "$MSETS")")")$([ -n "$MSETS" ] && [ -n "$FREE_B" ] && printf ', %s free' "$(human "$FREE_B")")$([ -n "$MSETS" ] && printf ')')"
    fi
    for n in $MT; do
      has=""; for x in $MSETS; do [ "$(set_field "$x" 2)" = "$n" ] && has=1; done
      [ -n "$has" ] || noloc="$noloc $n"
    done
    [ -n "$noloc" ] && echo "                     no local models for:$noloc (Device = Remote works without them)"
  fi
  tools_need remote_server "$PLUGINS" && echo "  6  Remote server   $(server_line "$REMOTE")"
  [ -n "$REMOTE_NOTE" ] && echo "                     $REMOTE_NOTE"
  if [ -n "$MXS" ]; then
    if [ -n "$MX_DIR" ]; then
      echo "  7  Matchbox        $MXS -> $MX_DIR/$MXS"
      [ -n "$MX_PRESETS" ] && echo "                     and the Matchbox browser of every Flame here: $(printf '%s' "$MX_PRESETS" | wc -l | tr -d ' ') x <presets>/matchbox/shaders/cwTools"
    else echo "  7  Matchbox        none ($MXS stays in the package: matchbox/)"; fi
  fi
  echo "     backups         $BACKUP_DIR"
  [ -n "$PROBLEMS" ] && printf '%s' "$PROBLEMS" | sed 's/^/  PROBLEM  /'
  return 0
}

plugins_install() {
  echo "cwTools installer, package $PKG_VERSION ($OS)"
  detect_flame
  echo

  # bundles in the package
  local BSRC n
  BSRC="$(abspath "${OPT_BUNDLES_FROM:-$HERE/bundles/$OS}")"
  [ -d "$BSRC" ] || die "no bundles at $BSRC (run install.sh from an unpacked cwTools package, or --bundles-from DIR)"
  AVAIL=""
  for n in $PLUGIN_NAMES_ALL; do [ -d "$BSRC/$n.ofx.bundle" ] && AVAIL="$AVAIL $n"; done
  AVAIL="${AVAIL# }"
  [ -n "$AVAIL" ] || die "no plugin bundles ($PLUGIN_NAMES_ALL) in $BSRC"
  echo "This package: $(for n in $AVAIL; do printf '%s %s  ' "$n" "$(tool_version "$n")"; done)"
  [ -n "$OPT_PLUGINS" ] && { opt_check --plugins "$OPT_PLUGINS" chk_plugins; OPT_PLUGINS="$VALID"; }

  # plan state: a value per line, and whether it was chosen (option or edit) or is a default
  local PLUGINS MSETS_ANS CH_PD CH_MM CH_SETS CH_REMOTE DM
  local OLD_MANIFEST OLD_PD OLD_TOOLS OLD_MM OLD_CONF STD_PLUGIN STD_ROOT ALL_TOOLS KEEP_TOOLS
  local MT MSETS PROBLEMS REMOTE_NOTE NEED_B FREE_B MXS MX_DIR CH_MX MX_PRESETS
  PLUGINS="${OPT_PLUGINS:-$AVAIL}"
  ROOT="$(abspath "${OPT_ROOT:-$DEF_ROOT}")"
  PLUGIN_DIR="$(abspath "$OPT_PLUGIN_DIR")"; CH_PD="${OPT_PLUGIN_DIR:+1}"
  MMODE="$OPT_MODELS"; CH_MM="${OPT_MODELS:+1}"
  MSETS_ANS="$OPT_MODEL_SETS"; CH_SETS="${OPT_MODEL_SETS:+1}"
  REMOTE="$OPT_REMOTE"; CH_REMOTE="$REMOTE_SET"
  MX_DIR="$(abspath "$OPT_MX")"; CH_MX="$MX_SET"
  while :; do
    plan_derive
    plan_show
    [ -n "$DRY" ] && echo "  (dry-run: nothing is changed)"
    if [ -n "$YES" ]; then
      [ -z "$PROBLEMS" ] || die "the plan has problems (see above)"
      echo "Install: yes (--yes)"; break
    fi
    if ! read_line "Install? [Enter = yes, a line number = change it, n = cancel]: "; then
      echo; REPLY_LINE=""; [ -z "$PROBLEMS" ] || die "the plan has problems (see above)"
    fi
    case "$(lc "$REPLY_LINE")" in
      ""|y|yes) [ -z "$PROBLEMS" ] && break
                echo "Fix the PROBLEM line(s) first: type the number of the line to change." ;;
      n|no|q) die "cancelled" ;;
      1) ask PLUGINS "Plugins: names or numbers ($(numbered_list "$AVAIL")), or all" "$PLUGINS" chk_plugins ;;
      2) ask ROOT "cwTools folder (models, cwtools.conf, install manifest + log)" "$ROOT" chk_folder ;;
      3) ask PLUGIN_DIR "OFX plugin folder (Flame's standard one: $DEF_PLUGIN_DIR)" "$PLUGIN_DIR" chk_folder; CH_PD=1 ;;
      4) if [ -z "$MT" ]; then echo "  (the chosen plugins have no models)"
         else
           echo "  download = from the release; from:FOLDER (or just the folder) = archives or model folders"
           echo "  there; shared:FOLDER = one studio folder all workstations use, nothing copied;"
           echo "  skip = no local models (Device = Remote only)"
           ask MMODE "Models" "$MMODE" chk_models_mode; CH_MM=1; CH_SETS=""
         fi ;;
      5) if [ -z "$MT" ] || [ "$MMODE" = skip ]; then echo "  (no models are installed: change line 4 first)"
         else
           list_models "$MT"
           ask MSETS_ANS "Model sets: names, default or all" "$MSETS" chk_sets; CH_SETS=1
         fi ;;
      6) if tools_need remote_server "$PLUGINS"; then
           ask REMOTE "Remote server (cwtools-infer, e.g. http://gpu-box:8201; - = none)" "$REMOTE" chk_url; CH_REMOTE=1
         else echo "  (not used by $PLUGINS)"; fi ;;
      7) if [ -n "$MXS" ]; then
           ask MX_DIR "Matchbox folder (- = none)" "$MX_DIR" chk_folder; CH_MX=1
         else echo "  (no Matchbox comes with $PLUGINS)"; fi ;;
      *) echo "  Enter, a line number (1-7) or n" ;;
    esac
  done

  # the SAM 3 licence is asked before anything is copied
  local x SAM_OK=""
  for x in $MSETS $(always_sets "$MT"); do
    [ "$(set_field "$x" 2)" = cwSegment ] || continue
    if sam_license_ok; then SAM_OK=y; else SAM_OK=n; fi
    break
  done

  # privileges: write test, sudo only where needed
  P_SUDO="$(sudo_for "$PLUGIN_DIR" $(for n in $PLUGINS cwProbe; do [ -e "$PLUGIN_DIR/$n.ofx.bundle" ] && echo "$PLUGIN_DIR/$n.ofx.bundle"; done))"
  R_SUDO="$(sudo_for "$ROOT")"
  BK_SUDO="$(sudo_for "$BACKUP_DIR")"
  local need=""
  [ -n "$P_SUDO" ] && need="$need $PLUGIN_DIR"
  [ -n "$R_SUDO" ] && need="$need $ROOT"
  [ -n "$BK_SUDO" ] && [ -z "$R_SUDO" ] && need="$need $BACKUP_DIR"
  MX_SUDO=""; [ -n "$MXS" ] && [ -n "$MX_DIR" ] && MX_SUDO="$(sudo_for "$MX_DIR")"
  [ -n "$MX_SUDO" ] && need="$need $MX_DIR"
  MXP_SUDO=""
  while IFS= read -r c; do [ -n "$c" ] && [ -n "$(sudo_for "$c")" ] && MXP_SUDO=sudo; done <<EOF
$MX_PRESETS
EOF
  [ -n "$MXP_SUDO" ] && need="$need /opt/Autodesk/presets/*/matchbox/shaders"
  ask_sudo "${need# }"
  start_log plan_show

  xr $P_SUDO mkdir -p "$PLUGIN_DIR"
  xr $R_SUDO mkdir -p "$ROOT"
  xr $BK_SUDO mkdir -p "$BACKUP_DIR"

  meta os "$OS"; meta mode plugins; meta plugin_dir "$PLUGIN_DIR"; meta root "$ROOT"
  meta backup_dir "$BACKUP_DIR"; meta installed "$(date '+%Y-%m-%d %H:%M:%S')"

  # dev-only cwProbe never stays in a plugin folder
  if [ -d "$PLUGIN_DIR/cwProbe.ofx.bundle" ]; then
    xr $(either "$P_SUDO" "$BK_SUDO") mv "$PLUGIN_DIR/cwProbe.ofx.bundle" "$BACKUP_DIR/cwProbe.ofx.bundle.dev-only-$STAMP"
    info "moved dev-only cwProbe out of the plugin folder: $BACKUP_DIR/cwProbe.ofx.bundle.dev-only-$STAMP"
  fi

  # bundles (safe while Flame runs: see put(); backups go to BACKUP_DIR, never the plugin folder)
  local SRC DST B d f SUF ov
  for n in $PLUGINS; do
    SRC="$BSRC/$n.ofx.bundle"; DST="$PLUGIN_DIR/$n.ofx.bundle"
    if [ -d "$DST" ] && diff -rq "$SRC" "$DST" >/dev/null 2>&1; then
      info "unchanged $DST"
    else
      if [ -d "$DST" ]; then
        ov="$(manifest_meta "$OLD_MANIFEST" "tool:$n" | tr -c 'A-Za-z0-9._+-' '_' | sed 's/_*$//')"
        SUF="${ov:+v$ov}"; SUF="${SUF:-pre-$STAMP}"
        B="$BACKUP_DIR/$n.ofx.bundle.$SUF"
        [ -e "$B" ] && B="$(backup_name "$B-$STAMP")"
        xr $BK_SUDO cp -pR "$DST" "$B" || die "backup failed: $B"
        info "backup  $B"
      fi
      while IFS= read -r d <&3; do xr $P_SUDO mkdir -p "$DST/${d#./}" || die "mkdir failed"; done 3< <(cd "$SRC" && find . -type d)
      while IFS= read -r f <&3; do put "$P_SUDO" "$SRC/${f#./}" "$DST/${f#./}" || die "copy failed: $DST/${f#./}"; done 3< <(cd "$SRC" && find . -type f)
      if [ -d "$DST" ]; then  # files that no longer exist in the new bundle
        while IFS= read -r f <&3; do [ -e "$SRC/${f#./}" ] || xr $P_SUDO rm -f "$DST/${f#./}"; done 3< <(cd "$DST" && find . -type f)
      fi
      if [ "$OS" = macos ]; then xr $P_SUDO xattr -dr com.apple.quarantine "$DST" 2>/dev/null; fi
      info "installed $n $(tool_version "$n") -> $DST"
    fi
    meta "tool:$n" "$(tool_version "$n")"
    rec bundle "$n" - "$DST"
    while IFS= read -r f <&3; do rec bundlefile "$n" "$(sha_of "$SRC/${f#./}")" "$DST/${f#./}"; done 3< <(cd "$SRC" && find . -type f | sort)
  done

  # never leave backup copies of our bundles where Flame scans for plugins
  for d in "$PLUGIN_DIR"/cw*.ofx.bundle.*; do
    [ -d "$d" ] || continue
    B="$BACKUP_DIR/$(basename "$d")"; [ -e "$B" ] && B="$(backup_name "$B.moved-$STAMP")"
    xr $(either "$P_SUDO" "$BK_SUDO") mv "$d" "$B"; info "moved stray backup bundle out of the plugin folder: $B"
  done

  # launcher for a non-standard plugin folder
  local LAUNCHER=""
  if [ -z "$STD_PLUGIN" ]; then
    LAUNCHER="$PLUGIN_DIR/launch_flame_cwtools.sh"
    write_launcher "$TMPD/launcher"
    if [ -f "$LAUNCHER" ] && cmp -s "$TMPD/launcher" "$LAUNCHER"; then :; else
      put "$P_SUDO" "$TMPD/launcher" "$LAUNCHER" && xr $P_SUDO chmod 755 "$LAUNCHER"
      info "launcher $LAUNCHER"
    fi
    rec launcher - "$(sha_of "$TMPD/launcher")" "$LAUNCHER"
    meta launcher "$LAUNCHER"
  fi

  # Matchboxes (xml + glsl side by side, one folder each)
  local mx
  if [ -n "$MXS" ] && [ -n "$MX_DIR" ]; then
    meta matchbox_dir "$MX_DIR"
    for mx in $MXS; do
      xr $MX_SUDO mkdir -p "$MX_DIR/$mx"
      for f in "$HERE/matchbox/$mx"/*; do
        [ -f "$f" ] || continue
        [ -f "$MX_DIR/$mx/$(basename "$f")" ] && cmp -s "$f" "$MX_DIR/$mx/$(basename "$f")" || put "$MX_SUDO" "$f" "$MX_DIR/$mx/$(basename "$f")"
        rec mxfile cwSegment "$(sha_of "$f")" "$MX_DIR/$mx/$(basename "$f")"
      done
      rec mxdir cwSegment - "$MX_DIR/$mx"
      info "matchbox $mx -> $MX_DIR/$mx"
      while IFS= read -r d; do
        [ -n "$d" ] || continue
        local s2; s2="$(sudo_for "$d")"
        xr $s2 mkdir -p "$d/cwTools"
        for f in "$HERE/matchbox/$mx"/*.xml "$HERE/matchbox/$mx"/*.glsl; do
          [ -f "$f" ] || continue
          [ -f "$d/cwTools/$(basename "$f")" ] && cmp -s "$f" "$d/cwTools/$(basename "$f")" || put "$s2" "$f" "$d/cwTools/$(basename "$f")"
          rec mxfile cwSegment "$(sha_of "$f")" "$d/cwTools/$(basename "$f")"
        done
        rec mxdir cwSegment - "$d/cwTools"
      done <<EOF
$MX_PRESETS
EOF
      [ -n "$MX_PRESETS" ] && info "matchbox $mx -> $(printf '%s' "$MX_PRESETS" | wc -l | tr -d ' ') Flame presets folders (<presets>/matchbox/shaders/cwTools)"
    done
  fi

  # models
  meta models_mode "$MMODE"; meta models_dir "$MODELS_DIR"
  for x in $MSETS; do
    n="$(set_field "$x" 2)"
    if [ "$n" = cwSegment ] && [ "$SAM_OK" != y ]; then
      warn "model set $x not installed (SAM License not accepted); cwSegment Remote mode still works"; continue
    fi
    case "$MMODE" in
      shared:*) shared_set "$x" "$MODELS_DIR/$n" "$n" ;;
      *) install_set "$R_SUDO" "$MMODE" "$x" "$MODELS_DIR/$n" "$n" ;;
    esac
  done
  for x in $(always_sets "$MT"); do
    in_list "$x" "$MSETS" && continue
    n="$(set_field "$x" 2)"
    if [ "$n" = cwSegment ] && [ "$SAM_OK" != y ]; then
      warn "model set $x not installed (SAM License not accepted): cwSegment text prompts need it"; continue
    fi
    install_always "$x"
  done
  case "$MMODE" in shared:*) ;; *) for n in $MT; do known_sums "$R_SUDO" "$MODELS_DIR/$n" "$n"; done ;; esac
  [ "$MMODE" = skip ] && [ -n "$MT" ] && info "no local models for $MT: Device = Remote (later: install.sh --models download|from:FOLDER)"

  # cwtools.conf at ROOT (the system default location) and, when the plugins would not find
  # that one on their own, next to the bundles
  update_conf "$R_SUDO" "$ROOT/cwtools.conf" "$ALL_TOOLS" "$OLD_MANIFEST"
  if [ -z "$STD_PLUGIN" ] || [ -z "$STD_ROOT" ]; then
    update_conf "$P_SUDO" "$PLUGIN_DIR/cwtools.conf" "$ALL_TOOLS" "$OLD_MANIFEST"
  fi

  # docs + manifest + VERSION
  for f in README.txt CHANGELOG.md THIRD_PARTY_NOTICES.txt $(cd "$HERE" && ls licenses/*.txt 2>/dev/null); do
    if [ -f "$HERE/$f" ]; then
      case "$f" in */*) xr $R_SUDO mkdir -p "$ROOT/$(dirname "$f")" ;; esac
      [ -f "$ROOT/$f" ] && cmp -s "$HERE/$f" "$ROOT/$f" || put "$R_SUDO" "$HERE/$f" "$ROOT/$f"
      rec doc - "$(sha_of "$HERE/$f")" "$ROOT/$f"
    fi
  done
  rec version - - "$ROOT/VERSION"
  rec log - - "$ROOT/install.log"
  finish_manifest "$R_SUDO" "$OLD_MANIFEST" "$OLD_MANIFEST" "$KEEP_TOOLS"
  write_version_file "$R_SUDO" "$MANIFEST_TMP" "$ROOT/VERSION"

  echo
  echo "Installed${DRY:+ (dry-run)}: $PLUGINS.$([ "$WARNINGS" -gt 0 ] && echo " $WARNINGS warning(s), see above.")"
  echo "  plugins  $PLUGIN_DIR  ($(manifest_tools "$MANIFEST_TMP"))"
  [ -f "$ROOT/cwtools.conf" ] || [ -n "$DRY" ] && echo "  config   $ROOT/cwtools.conf"
  echo "  check    $HERE/install.sh --check --root \"$ROOT\""
  [ -n "$MXS" ] && [ -n "$MX_DIR" ] && echo "  matchbox $MXS: Batch -> Matchbox -> cwTools folder (or $MX_DIR) -> $MXS.xml"
  if [ -n "$LAUNCHER" ]; then
    echo
    echo "  $PLUGIN_DIR is not Flame's standard OFX folder. Start Flame with"
    echo "    $LAUNCHER [flame install]"
    echo "  or set OFX_PLUGIN_PATH=$PLUGIN_DIR in Flame's environment."
  fi
  echo "  Restart Flame to load the new plugins (Batch -> OpenFX -> cwTools)."
  finish_log "$R_SUDO" "$ROOT"
}

write_launcher() { # OUT
  cat > "$1" <<'EOF'
#!/usr/bin/env bash
# Start Flame with the cwTools plugins in this folder (OFX_PLUGIN_PATH = this folder).
# Written by the cwTools installer because this is not Flame's standard OFX folder.
#   ./launch_flame_cwtools.sh [flame install]
#     Linux: /opt/Autodesk/flame_<version> (default: the newest one)
#     macOS: /Applications/Autodesk/Flame <version>.app (default: the newest one)
# CWTOOLS_DEBUG=1 ./launch_flame_cwtools.sh  also turns on verbose plugin logs
# (/tmp/cwSegment.log, /tmp/cwRefine.log, /tmp/cwMatte.log) and disables Flame's OFX
# plugin cache.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OFX_PLUGIN_PATH="$HERE${OFX_PLUGIN_PATH:+:$OFX_PLUGIN_PATH}"
if [ "${CWTOOLS_DEBUG:-0}" = 1 ]; then
  export DL_OFX_PLUGIN_CACHE_ENABLED=0 CWSEGMENT_VERBOSE=1 CWREFINE_VERBOSE=1 CWMATTE_VERBOSE=1
fi
case "$(uname -s)" in
  Linux)
    FLAME="${1:-$(ls -d /opt/Autodesk/flame_* 2>/dev/null | sort -V | tail -1)}"
    START="$FLAME/bin/startApplication" ;;
  Darwin)
    FLAME="${1:-$(ls -d /Applications/Autodesk/Flame*.app /Applications/Autodesk/*/Flame*.app 2>/dev/null | sort | tail -1)}"
    EXE="$(/usr/bin/defaults read "$FLAME/Contents/Info" CFBundleExecutable 2>/dev/null || true)"
    START="$FLAME/Contents/MacOS/$EXE" ;;
  *) echo "unsupported OS" >&2; exit 1 ;;
esac
[ -n "$FLAME" ] && [ -x "$START" ] || { echo "no Flame found (give the install folder as argument)" >&2; exit 1; }
echo "Flame:           $FLAME"
echo "OFX_PLUGIN_PATH: $OFX_PLUGIN_PATH"
exec "$START"
EOF
}

# ═══════════════════════════════════════════════════════════ check ══
plugins_check() {
  ROOT="$(abspath "${OPT_ROOT:-$DEF_ROOT}")"
  local M="$ROOT/$MANIFEST_NAME"
  [ -f "$M" ] || die "no install manifest at $M (use --root)"
  local pd tools n v; pd="$(manifest_meta "$M" plugin_dir)"; tools="$(manifest_tools "$M")"
  if [ -n "$OPT_PLUGINS" ]; then
    for n in $OPT_PLUGINS; do in_list "$n" "$tools" || die "$n is not installed here (installed: $tools)"; done
    tools="$OPT_PLUGINS"
  fi
  echo "cwTools install at $ROOT ($(tail -1 "$ROOT/VERSION" 2>/dev/null))"
  echo "  plugin folder  $pd"
  local mm; mm="$(manifest_meta "$M" models_mode)"
  [ -n "$mm" ] && echo "  models         $mm  ($(manifest_meta "$M" models_dir); every file is checked against its sha256 below)"
  for n in $tools; do
    v="$(grep -A1 CFBundleShortVersionString "$pd/$n.ofx.bundle/Contents/Info.plist" 2>/dev/null | sed -n 's/.*<string>\(.*\)<\/string>.*/\1/p' | head -1)"
    echo "  $n  installed $(manifest_meta "$M" "tool:$n")  (bundle ${v:-?})$(in_list "$n" "$PLUGIN_NAMES_ALL" && [ -f "$HERE/TOOLS" ] && awk -F'\t' -v n="$n" '$1==n {printf "  this package: %s", $2}' "$HERE/TOOLS")"
  done
  detect_flame
  echo "Files:"
  check_manifest_files "$M" "$tools"
  echo "  $CHK_OK files OK, $CHK_BAD problem(s)"
  local d
  for d in "$pd"/cw*.ofx.bundle.*; do [ -e "$d" ] && { echo "  STRAY    $d (Flame loads it: move it out of the plugin folder)"; CHK_BAD=$((CHK_BAD + 1)); }; done
  local conf=""
  if [ -n "${CWTOOLS_CONFIG:-}" ]; then conf="$CWTOOLS_CONFIG"
  elif [ -f "$pd/cwtools.conf" ]; then conf="$pd/cwtools.conf"
  elif [ "$OS" = linux ]; then conf=/opt/cwTools/cwtools.conf
  else conf="/Library/Application Support/cwTools/cwtools.conf"; fi
  local needs=""; for n in models_dir remote_server ort_lib; do tools_need "$n" "$tools" && needs=1; done
  if [ -n "$needs" ]; then
    echo "Config the plugins will read: $conf"
    if [ -f "$conf" ]; then grep -v '^[[:space:]]*#' "$conf" | grep -v '^[[:space:]]*$' | sed 's/^/  /'
    else echo "  (missing: plugins use built-in defaults)"; fi
    if [ "$conf" != "$ROOT/cwtools.conf" ] && [ -f "$ROOT/cwtools.conf" ] && [ -f "$conf" ] && ! cmp -s "$conf" "$ROOT/cwtools.conf"; then
      echo "  note: $ROOT/cwtools.conf differs from it (the plugins do not read that one)"
    fi
    local md; md="$(conf_get "$conf" models_dir)"
    for n in $tools; do
      tool_has_models "$n" && [ -n "$md" ] && ! ls "$md/$n"/*.onnx >/dev/null 2>&1 && echo "  note: no local models for $n in $md/$n (Device = Remote only)"
    done
    local u
    tools_need remote_server "$tools" && u="$(conf_get "$conf" remote_server)" && [ -n "$u" ] && { echo "Remote server:"; url_check "$u"; }
  fi
  [ "$CHK_BAD" -eq 0 ] && echo "CHECK OK" || { echo "CHECK FAILED ($CHK_BAD problem(s))"; return 1; }
}

# ═══════════════════════════════════════════════════════════ uninstall ══
plugins_uninstall() {
  ROOT="$(abspath "${OPT_ROOT:-$DEF_ROOT}")"
  local M="$ROOT/$MANIFEST_NAME"
  [ -f "$M" ] || die "no install manifest at $M (use --root)"
  local pd bd installed rm_tools left="" n
  pd="$(manifest_meta "$M" plugin_dir)"; bd="$(manifest_meta "$M" backup_dir)"; installed="$(manifest_tools "$M")"
  rm_tools="${OPT_PLUGINS:-$installed}"
  for n in $rm_tools; do in_list "$n" "$installed" || die "$n is not installed in $ROOT (installed: ${installed:-none})"; done
  for n in $installed; do in_list "$n" "$rm_tools" || left="$left $n"; done
  left="${left# }"
  local k t h p list="$TMPD/remove"; : > "$list"
  : > "$MANIFEST_TMP"
  while IFS="$TAB" read -r k t h p <&3; do
    local mine=""; in_list "$t" "$rm_tools" && mine=1
    [ "$k" = meta ] && case "$t" in tool:*) in_list "${t#tool:}" "$rm_tools" && mine=1 ;; esac
    if [ -n "$mine" ]; then
      case "$k" in
        bundle) case "$p" in "$pd"/*.ofx.bundle) echo "$p" >> "$list" ;; *) warn "skipping unexpected bundle path $p" ;; esac ;;
        mxdir) case "$p" in */cwExtract|*/matchbox/shaders/cwTools) [ -d "$p" ] && echo "$p" >> "$list" ;; *) warn "skipping unexpected Matchbox path $p" ;; esac ;;
        model|modelsums)
          if [ -n "$PURGE_MODELS" ]; then [ -e "$p" ] && echo "$p" >> "$list"; continue; fi ;;
      esac
      # entries of removed tools leave the manifest, except kept models (for a later --purge-models)
      case "$k" in model|modelsums) printf '%s\t%s\t%s\t%s\n' "$k" "$t" "$h" "$p" >> "$MANIFEST_TMP" ;; esac
      continue
    fi
    if [ -z "$left" ] && [ "$t" = - ]; then # last tool: shared files too
      case "$k" in
        launcher|version|doc|log) [ -e "$p" ] && echo "$p" >> "$list" ;;
        conf)
          if [ -f "$p" ]; then
            if [ "$h" != - ] && [ "$(sha_of "$p")" = "$h" ]; then echo "$p" >> "$list"
            else echo "keeping $p (edited since install, or not written by the installer)"; fi
          fi ;;
      esac
      continue
    fi
    if [ -z "$left" ] && { [ "$k" = model ] || [ "$k" = modelsums ]; }; then
      if [ -n "$PURGE_MODELS" ]; then [ -e "$p" ] && echo "$p" >> "$list"; continue; fi
    fi
    printf '%s\t%s\t%s\t%s\n' "$k" "$t" "$h" "$p" >> "$MANIFEST_TMP"
  done 3< "$M"
  local drop_manifest=""
  if [ -z "$left" ] && ! grep -q "^model" "$MANIFEST_TMP"; then drop_manifest=1; echo "$M" >> "$list"; fi
  echo "Uninstall: $rm_tools$([ -n "$left" ] && echo "  (stays installed: $left)")"
  echo "Will remove:"; sed 's/^/  /' "$list"
  [ -z "$PURGE_MODELS" ] && echo "Models stay in $ROOT/models (add --purge-models to remove them)."
  [ -d "$bd" ] && echo "Backups stay in $bd."
  confirm "Uninstall?" n || die "cancelled"
  P_SUDO="$(sudo_for "$pd")"; R_SUDO="$(sudo_for "$ROOT")"
  local need=""; [ -n "$P_SUDO" ] && need="$pd"; [ -n "$R_SUDO" ] && need="$need $ROOT"
  ask_sudo "${need# }"
  while IFS= read -r p <&3; do
    case "$p" in "$pd"/*) xr $P_SUDO rm -rf "$p" ;; "$ROOT"/*) xr $R_SUDO rm -rf "$p" ;; *) xr $(sudo_for "$(dirname "$p")") rm -rf "$p" ;; esac
  done 3< "$list"
  if [ -z "$drop_manifest" ]; then
    put "$R_SUDO" "$MANIFEST_TMP" "$M"
    if [ -n "$left" ]; then write_version_file "$R_SUDO" "$MANIFEST_TMP" "$ROOT/VERSION"; fi
  fi
  # empty folders we created
  for p in "$ROOT"/models/* "$ROOT/models" "$ROOT/licenses" "$ROOT"; do
    [ -d "$p" ] && xr $R_SUDO rmdir "$p" 2>/dev/null
  done
  echo "Uninstalled${DRY:+ (dry-run)}: ${rm_tools:-models only}."
}

# ═══════════════════════════════════════════════════════════ server ══
first_ipv4() {
  local a=""
  if command -v ip >/dev/null 2>&1; then
    a="$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $4}' | head -1 | cut -d/ -f1)"
  fi
  [ -n "$a" ] || a="$(hostname -I 2>/dev/null | awk '{print $1}')"
  printf '%s' "${a:-127.0.0.1}"
}

is_loopback() { case "$1" in 127.*|localhost|::1) return 0 ;; esac; return 1; }
only_loopback() { # "CIDR,CIDR" -> 0 when every entry is loopback
  local e
  for e in $(printf '%s' "$1" | tr ',' ' '); do case "$e" in 127.*|::1|::1/128) ;; *) return 1 ;; esac; done
}
local_subnet() { # ADDR (0.0.0.0 = the first address) -> its network as CIDR, e.g. 192.0.2.0/24
  local a="$1" c ip pre o1 o2 o3 o4 n m
  [ "$a" = 0.0.0.0 ] && a="$(first_ipv4)"
  command -v ip >/dev/null 2>&1 || return 0
  c="$(ip -o -4 addr show 2>/dev/null | awk -v a="$a" '{split($4, x, "/"); if (x[1] == a) {print $4; exit}}')"
  [ -n "$c" ] || return 0
  ip="${c%/*}"; pre="${c#*/}"
  IFS=. read -r o1 o2 o3 o4 <<< "$ip"
  n=$(( (o1 << 24) | (o2 << 16) | (o3 << 8) | o4 ))
  m=$(( pre == 0 ? 0 : (0xFFFFFFFF << (32 - pre)) & 0xFFFFFFFF ))
  n=$(( n & m ))
  printf '%d.%d.%d.%d/%s' $(( (n >> 24) & 255 )) $(( (n >> 16) & 255 )) $(( (n >> 8) & 255 )) $(( n & 255 )) "$pre"
}

find_python() {
  local p
  for p in ${OPT_PYTHON:-python3.13 python3.12 python3.11 python3.10 python3}; do
    command -v "$p" >/dev/null 2>&1 || [ -x "$p" ] || continue
    if "$p" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$p" 2>/dev/null || echo "$p"; return 0
    fi
  done
  return 1
}

SERVER_FILES="cwinfer_server.py cwtensor.py canary.py netallow.py make_canary.py client_test.py requirements.txt API.md"

server_install() {
  [ "$OS" = linux ] || die "--server is Linux only (the Remote inference server needs an NVIDIA GPU + CUDA on Linux)"
  echo "cwTools $PKG_VERSION: Remote inference server (cwtools-infer) setup"
  local SSRC="$HERE/server" f
  for f in $SERVER_FILES; do [ -f "$SSRC/$f" ] || die "server file missing in the package: $SSRC/$f"; done
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' \
    || warn "nvidia-smi not found: the server needs an NVIDIA GPU + driver"

  if [ -n "$OPT_ROOT" ]; then SROOT="$OPT_ROOT"; else ask SROOT "Server folder" "$DEF_SERVER_ROOT"; fi
  SROOT="$(abspath "$SROOT")"
  local OLD_ENV="$SROOT/cwtools-infer.env" OLD_MANIFEST="$SROOT/$MANIFEST_NAME"
  local PY; PY="$(find_python)" || die "python >= 3.10 with venv/ensurepip not found (install python3.11 + python3.11-venv, or --python PATH)"
  info "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
  local SVC_USER="${OPT_SVC_USER:-${SUDO_USER:-$(id -un)}}"
  local SERVICE UNIT LISTEN PORT ALLOW MMODE=skip MSETS=""
  if [ -n "$OPT_SERVICE" ]; then SERVICE="$OPT_SERVICE"; else ask SERVICE "systemd unit: system (starts at boot, needs sudo) | user | none" system chk_service; fi
  UNIT="${OPT_UNIT:-$(manifest_meta "$OLD_MANIFEST" unit)}"; UNIT="${UNIT:-cwtools-infer}"
  if [ "$SERVICE" = system ] && [ -z "$OPT_SVC_USER" ]; then ask SVC_USER "Run the server as user" "$SVC_USER" chk_user; fi
  local dl; dl="$(conf_get "$OLD_ENV" CWINFER_HOST)"
  if [ -n "$OPT_LISTEN" ]; then LISTEN="$OPT_LISTEN"; else ask LISTEN "Listen address" "${dl:-$(first_ipv4)}"; fi
  dl="$(conf_get "$OLD_ENV" CWINFER_PORT)"
  if [ -n "$OPT_PORT" ]; then PORT="$OPT_PORT"; else ask PORT "Port" "${dl:-8201}" chk_port; fi
  # allowed clients (CWINFER_ALLOW). An env file from a pre-release install has no such key:
  # treat it like a fresh install and say so, since the old server allowed a fixed range.
  local da=""
  if [ -f "$OLD_ENV" ]; then
    if conf_has "$OLD_ENV" CWINFER_ALLOW; then da="$(conf_get "$OLD_ENV" CWINFER_ALLOW)"
    else warn "existing server settings have no client allow-list yet; choose which clients may connect"; fi
  fi
  if [ -n "$ALLOW_SET" ]; then ALLOW="$OPT_ALLOW"
  else
    if [ -z "$da" ] && ! is_loopback "$LISTEN"; then
      local sn; sn="$(local_subnet "$LISTEN")"
      da="$LOOPBACK_ALLOW"
      [ -n "$sn" ] && yesno "Allow clients from the local network $sn? (the server has no authentication)" y && da="$da,$sn"
    fi
    echo "Clients allowed to connect: comma-separated addresses/networks (CIDR). No authentication:"
    echo "keep this to your LAN or VPN."
    ask ALLOW "Allowed clients" "${da:-$LOOPBACK_ALLOW}" chk_allow
  fi
  ALLOW="$(printf '%s' "$ALLOW" | tr -d ' ')"; [ -n "$ALLOW" ] || ALLOW="$LOOPBACK_ALLOW"
  case "$ALLOW" in *[!0-9A-Fa-f.:/,]*) die "--allow: comma-separated IPs or CIDRs only (got: $ALLOW)" ;; esac
  if ! is_loopback "$LISTEN" && only_loopback "$ALLOW"; then
    warn "listening on $LISTEN but only this machine is allowed ($ALLOW): other workstations get HTTP 403 (--allow)"
  fi
  # Listening on one LAN address: this machine reaches the server from that address, not
  # from loopback, so allow it too (else the installer's own health check / client test and
  # any local client get HTTP 403). 0.0.0.0 / :: are reached via loopback.
  case "$LISTEN" in 0.0.0.0|::|"") ;; *)
    local le; le="$(printf '%s' "$LISTEN" | sed 's/[.]/\\./g')"
    if ! is_loopback "$LISTEN" && ! printf ',%s,' "$ALLOW" | grep -q ",$le\(/32\|/128\)\?,"; then
      case "$LISTEN" in *:*) ALLOW="$ALLOW,$LISTEN/128" ;; *) ALLOW="$ALLOW,$LISTEN/32" ;; esac
      info "allowed clients: added this machine's listen address ($LISTEN)"
    fi ;;
  esac
  if [ -n "$OPT_MODELS" ]; then MMODE="$OPT_MODELS"
  else
    local dm=download; [ -d "$HERE/server_models" ] && dm="from:$HERE/server_models"
    ask MMODE "Models: download | from:FOLDER | skip" "$dm" chk_models_mode; [ -n "$MMODE" ] || MMODE=skip
  fi
  if [ "$MMODE" != skip ]; then
    local MT=server
    if [ -n "$OPT_MODEL_SETS" ]; then pick_sets server "$OPT_MODEL_SETS" || die "--model-sets: see above"
    elif [ -n "$YES" ]; then pick_sets server default
    else
      list_models server
      local ans; ask ans "Model sets (names, 'default' or 'all')" "$(sets_for server default)" chk_sets
      pick_sets server "$ans"
    fi
  fi
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then
    [ -f "$OLD_MANIFEST" ] || warn "port $PORT is already in use on this machine"
  fi

  S_SUDO=""; [ "$(id -un)" = "$SVC_USER" ] || S_SUDO=sudo
  local MKROOT_SUDO; MKROOT_SUDO="$(sudo_for "$SROOT")"
  echo
  echo "Plan:"
  echo "  server folder  $SROOT  (owner $SVC_USER: server/ venv/ models/ canary/ logs/ cwtools-infer.env)"
  echo "  python         $PY -> venv, pip install -r requirements.txt (onnxruntime-gpu 1.22.0 + CUDA wheels)"
  echo "  models         $MMODE$([ -n "$MSETS" ] && printf ': %s (%s)' "$MSETS" "$(human "$(sets_total "$MSETS")")")"
  echo "  listen         http://$LISTEN:$PORT"
  echo "  allowed        $ALLOW"
  echo "  systemd        $SERVICE unit $UNIT"
  confirm "Install?" y || die "cancelled"
  local need=""
  [ -n "$MKROOT_SUDO" ] && need="$SROOT"
  [ "$SERVICE" = system ] && [ "$(id -u)" != 0 ] && need="$need /etc/systemd/system"
  ask_sudo "${need# }"
  start_log

  # the server folder belongs to the service user; everything below runs without sudo
  # (as that user via sudo -u when it is not you)
  if [ ! -d "$SROOT" ] || [ -n "$MKROOT_SUDO" ]; then
    xr $MKROOT_SUDO mkdir -p "$SROOT"
    [ -n "$MKROOT_SUDO" ] && xr sudo chown "$SVC_USER:$(id -gn "$SVC_USER")" "$SROOT"
  fi
  local AS=""; [ -n "$S_SUDO" ] && AS="sudo -u $SVC_USER"
  xr $AS mkdir -p "$SROOT/server" "$SROOT/models" "$SROOT/canary" "$SROOT/logs"
  meta os linux; meta mode server; meta "tool:server" "$PKG_VERSION"; meta root "$SROOT"; meta service "$SERVICE"
  meta unit "$UNIT"; meta service_user "$SVC_USER"; meta installed "$(date '+%Y-%m-%d %H:%M:%S')"

  for f in $SERVER_FILES; do put "$AS" "$SSRC/$f" "$SROOT/server/$f" || die "copy failed: $f"; rec serverfile server "$(sha_of "$SSRC/$f")" "$SROOT/server/$f"; done
  if [ -f "$HERE/THIRD_PARTY_NOTICES.txt" ]; then # model licences (server/ is removed whole on uninstall)
    xr $AS mkdir -p "$SROOT/server/licenses"
    for f in THIRD_PARTY_NOTICES.txt $(cd "$HERE" && ls licenses/*.txt 2>/dev/null); do
      put "$AS" "$HERE/$f" "$SROOT/server/$f" && rec serverfile server "$(sha_of "$HERE/$f")" "$SROOT/server/$f"
    done
  fi
  info "server files -> $SROOT/server"

  # venv
  if [ ! -x "$SROOT/venv/bin/python" ]; then
    info "creating venv $SROOT/venv"
    xr $AS "$PY" -m venv "$SROOT/venv" || die "venv creation failed"
  fi
  info "pip install -r requirements.txt (may download ~1.5 GB of CUDA wheels the first time)"
  xr $AS "$SROOT/venv/bin/python" -m pip install -q --disable-pip-version-check -r "$SROOT/server/requirements.txt" || die "pip install failed"
  rec venv server - "$SROOT/venv"
  if [ -z "$DRY" ]; then
    local prov
    prov="$($AS "$SROOT/venv/bin/python" -c 'import onnxruntime as ort; ort.preload_dlls(); print(ort.__version__, ",".join(ort.get_available_providers()))' 2>&1 | tail -1)"
    info "onnxruntime: $prov"
    case "$prov" in *CUDAExecutionProvider*) ;; *) warn "CUDAExecutionProvider not available in the venv" ;; esac
  fi

  # models
  local x
  MODEL_KIND=model
  for x in $MSETS; do install_set "$AS" "$MMODE" "$x" "$SROOT/models/$(set_field "$x" 3)" server; done
  [ -n "$MSETS" ] || warn "no models installed (later: install.sh --server --models download)"

  # canaries (reference outputs the server checks at every model load)
  local ENVV="CWINFER_MODELS=$SROOT/models CWINFER_CANARY=$SROOT/canary CWINFER_LOG_DIR=$SROOT/logs"
  if ls "$SROOT"/models/*/*.onnx >/dev/null 2>&1 || [ -n "$DRY" ]; then
    info "building canary references for models without one (needs the GPU for a minute)"
    ( cd "$SROOT/server" && xr $AS env $ENVV "$SROOT/venv/bin/python" make_canary.py $([ -z "$CANARY_CPU" ] && echo --no-cpu) ) \
      || warn "canary build failed: those models load without a canary check"
  fi
  # startup canary = the server's default models if installed with a reference, else the first one
  local startup="" cand m sum stem
  for m in "$SROOT"/models/*/*.onnx; do
    [ -f "$m" ] || continue
    stem="$(basename "$m" .onnx)"
    sum="$(awk -v n="$(basename "$m")" '{x=$0; sub(/^[0-9a-fA-F]+[ \t]+\*?/, "", x); if (x==n) {print tolower($1); exit}}' "$(dirname "$m")/SHA256SUMS" 2>/dev/null)"
    [ -n "$sum" ] && [ -f "$SROOT/canary/$stem.$(printf '%s' "$sum" | cut -c1-16).npz" ] || continue
    case "$stem" in vitmatte_core|BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs) startup="$startup,$stem" ;; *) [ -z "$cand" ] && cand="$stem" ;; esac
  done
  startup="${startup#,}"; [ -n "$startup" ] || startup="$cand"
  [ -n "$startup" ] || [ -n "$DRY" ] || warn "no model with a canary reference: no startup integrity check"

  # environment file
  cat > "$TMPD/env" <<EOF
# cwtools-infer settings (read by the systemd unit $UNIT). Restart the unit after editing.
CWINFER_HOST=$LISTEN
CWINFER_PORT=$PORT
# clients allowed to connect (comma-separated IPs / CIDRs; others get HTTP 403). The server has
# no authentication: keep this to your LAN or VPN.
CWINFER_ALLOW=$ALLOW
CWINFER_MODELS=$SROOT/models
CWINFER_CANARY=$SROOT/canary
CWINFER_LOG_DIR=$SROOT/logs
# models loaded + canary-checked at startup (comma-separated names; empty = none)
CWINFER_STARTUP_CANARY=$startup
# release a model after this many idle seconds; at most this many loaded at once
CWINFER_IDLE_S=600
CWINFER_MAX_SESSIONS=3
CWINFER_MAX_QUEUE=8
CWINFER_SHRINK=1
CWINFER_GPU_MEM_LIMIT_GB=48
CWINFER_MAX_BODY_MB=512
PYTHONUNBUFFERED=1
EOF
  BACKUP_DIR="$SROOT/backups"; BK_SUDO="$AS"
  # upgrade: an env file without CWINFER_ALLOW gets the key appended even when the rest of it
  # is kept (a server without it would refuse every remote client)
  if [ -f "$OLD_ENV" ] && ! conf_has "$OLD_ENV" CWINFER_ALLOW; then
    xr $AS mkdir -p "$BACKUP_DIR"
    local bn; bn="$(backup_name "$BACKUP_DIR/cwtools-infer.env.$STAMP")"
    xr $AS cp "$OLD_ENV" "$bn" && info "backup  $bn"
    { cat "$OLD_ENV"; printf '%s\n' "# clients allowed to connect (added by the installer; no authentication: LAN / VPN only)" "CWINFER_ALLOW=$ALLOW"; } > "$TMPD/env.old"
    put "$AS" "$TMPD/env.old" "$OLD_ENV" && info "added CWINFER_ALLOW=$ALLOW to $OLD_ENV"
  fi
  install_conf_file "$AS" "$TMPD/env" "$SROOT/cwtools-infer.env" "cwtools-infer.env"
  case "$CONF_RESULT" in
    written) rec envfile server "$(sha_of "$TMPD/env")" "$SROOT/cwtools-infer.env" ;;
    *) rec envfile server "$(manifest_get "$OLD_MANIFEST" envfile "$SROOT/cwtools-infer.env" | grep . || echo -)" "$SROOT/cwtools-infer.env" ;;
  esac

  # systemd unit
  local UNIT_FILE="" SC="" HEALTH_HOST="$LISTEN"
  [ "$LISTEN" = 0.0.0.0 ] && HEALTH_HOST=127.0.0.1
  if [ "$SERVICE" != none ]; then
    if [ "$SERVICE" = system ]; then UNIT_FILE="/etc/systemd/system/$UNIT.service"; SC="sudo systemctl"; [ "$(id -u)" = 0 ] && SC=systemctl
    else UNIT_FILE="$HOME/.config/systemd/user/$UNIT.service"; SC="systemctl --user"; fi
    {
      echo "[Unit]"
      echo "Description=cwTools remote inference server (cwtools-infer :$PORT)"
      echo "After=network-online.target"
      echo "Wants=network-online.target"
      echo
      echo "[Service]"
      echo "Type=simple"
      [ "$SERVICE" = system ] && echo "User=$SVC_USER"
      echo "WorkingDirectory=$SROOT/server"
      echo "EnvironmentFile=$SROOT/cwtools-infer.env"
      echo "ExecStart=$SROOT/venv/bin/python $SROOT/server/cwinfer_server.py"
      echo "Restart=on-failure"
      echo "RestartSec=10"
      echo "# exit 2 = startup integrity check failed (sha / canary): do not restart in a loop"
      echo "RestartPreventExitStatus=2"
      echo
      echo "[Install]"
      [ "$SERVICE" = system ] && echo "WantedBy=multi-user.target" || echo "WantedBy=default.target"
    } > "$TMPD/unit"
    local US=""; [ "$SERVICE" = system ] && [ "$(id -u)" != 0 ] && US=sudo
    xr $US mkdir -p "$(dirname "$UNIT_FILE")"
    put "$US" "$TMPD/unit" "$UNIT_FILE" && xr $US chmod 644 "$UNIT_FILE"
    rec unit server "$(sha_of "$TMPD/unit")" "$UNIT_FILE"
    meta unit_file "$UNIT_FILE"
    info "unit    $UNIT_FILE"
    xr $SC daemon-reload
    if confirm "Enable $UNIT at boot and (re)start it now?" y; then
      xr $SC enable "$UNIT" >/dev/null 2>&1
      xr $SC restart "$UNIT" || warn "start failed: $SC status $UNIT"
      wait_health "http://$HEALTH_HOST:$PORT" && server_client_test "http://$HEALTH_HOST:$PORT"
    fi
  fi

  printf '%s\ninstalled %s: server\n' "$PKG_VERSION" "$(date '+%Y-%m-%d %H:%M')" > "$TMPD/VERSION"
  put "$AS" "$TMPD/VERSION" "$SROOT/VERSION"; rec version server - "$SROOT/VERSION"
  finish_manifest "$AS" "$OLD_MANIFEST" "$OLD_MANIFEST" ""

  echo
  echo "cwtools-infer $PKG_VERSION installed in $SROOT${DRY:+ (dry-run)}.$([ "$WARNINGS" -gt 0 ] && echo " $WARNINGS warning(s), see above.")"
  echo "  URL for the plugins (remote_server in cwtools.conf):  http://$LISTEN:$PORT"
  if [ "$SERVICE" = none ]; then
    echo "  run in the foreground:"
    echo "    set -a; . $SROOT/cwtools-infer.env; set +a; $SROOT/venv/bin/python $SROOT/server/cwinfer_server.py"
  else
    echo "  start / stop / status:  $SC start|stop|status $UNIT"
    if [ "$SERVICE" = system ]; then echo "  log:  journalctl -u $UNIT -f    or  tail -f $SROOT/logs/server_$PORT.log"
    else
      echo "  log:  journalctl --user -u $UNIT -f    or  tail -f $SROOT/logs/server_$PORT.log"
      echo "  A user unit runs only while you are logged in, unless: sudo loginctl enable-linger $(id -un)"
    fi
  fi
  echo "  settings:  $SROOT/cwtools-infer.env;  check: $HERE/install.sh --server --check --root \"$SROOT\""
  finish_log "$AS" "$SROOT"
}

wait_health() { # URL
  [ -n "$DRY" ] && return 0
  local i
  info "waiting for $1/health (the startup canary loads models) ..."
  for i in $(seq 1 90); do
    if curl -s -m 3 -o /dev/null -w '%{http_code}' "$1/health" 2>/dev/null | grep -q 200; then
      info "health OK: $(curl -s -m 5 "$1/health" | head -c 300)"; echo; return 0
    fi
    sleep 2
  done
  warn "no answer from $1/health after 180 s (see the log in $SROOT/logs)"
  return 1
}

server_client_test() { # URL
  [ -n "$NO_TEST" ] && return 0
  [ -n "$DRY" ] && return 0
  if ! ls "$SROOT"/models/*/vitmatte_core.onnx >/dev/null 2>&1; then
    info "client test skipped (needs the vitmatte_core model)"; return 0
  fi
  info "client test: $SROOT/server/client_test.py $1"
  ( cd "$SROOT/server" && "$SROOT/venv/bin/python" client_test.py "$1" ) || warn "client test FAILED"
}

server_check() {
  SROOT="$(abspath "${OPT_ROOT:-$DEF_SERVER_ROOT}")"
  local M="$SROOT/$MANIFEST_NAME"
  [ -f "$M" ] || die "no install manifest at $M (use --root)"
  echo "cwtools-infer at $SROOT: version $(head -1 "$SROOT/VERSION" 2>/dev/null), this package $PKG_VERSION"
  echo "Files:"; check_manifest_files "$M"; echo "  $CHK_OK files OK, $CHK_BAD problem(s)"
  local svc unit; svc="$(manifest_meta "$M" service)"; unit="$(manifest_meta "$M" unit)"
  case "$svc" in
    system) systemctl status "$unit" --no-pager 2>/dev/null | head -5 ;;
    user) systemctl --user status "$unit" --no-pager 2>/dev/null | head -5 ;;
  esac
  local h p; h="$(conf_get "$SROOT/cwtools-infer.env" CWINFER_HOST)"; p="$(conf_get "$SROOT/cwtools-infer.env" CWINFER_PORT)"
  [ "$h" = 0.0.0.0 ] && h=127.0.0.1
  echo "Server:"; url_check "http://$h:$p" || CHK_BAD=$((CHK_BAD + 1))
  [ "$CHK_BAD" -eq 0 ] && echo "CHECK OK" || { echo "CHECK FAILED ($CHK_BAD problem(s))"; return 1; }
}

server_uninstall() {
  SROOT="$(abspath "${OPT_ROOT:-$DEF_SERVER_ROOT}")"
  local M="$SROOT/$MANIFEST_NAME"
  [ -f "$M" ] || die "no install manifest at $M (use --root)"
  local svc unit uf su; svc="$(manifest_meta "$M" service)"; unit="$(manifest_meta "$M" unit)"
  uf="$(manifest_meta "$M" unit_file)"; su="$(manifest_meta "$M" service_user)"
  echo "Will remove: unit ${uf:-none}, $SROOT/{server,venv,canary,VERSION,$MANIFEST_NAME}$([ -n "$PURGE_MODELS" ] && echo ", models")"
  echo "Keeps: $SROOT/cwtools-infer.env if edited, logs/, backups/$([ -z "$PURGE_MODELS" ] && echo ", models/ (--purge-models removes them)")"
  confirm "Uninstall cwtools-infer from $SROOT?" n || die "cancelled"
  local SC="" US=""
  if [ "$svc" = system ]; then SC="sudo systemctl"; US=sudo; [ "$(id -u)" = 0 ] && { SC=systemctl; US=""; }; fi
  [ "$svc" = user ] && SC="systemctl --user"
  if [ -n "$SC" ] && [ -n "$unit" ]; then
    xr $SC disable --now "$unit" >/dev/null 2>&1
    [ -n "$uf" ] && [ -f "$uf" ] && xr $US rm -f "$uf"
    xr $SC daemon-reload
  fi
  local AS=""; [ -n "$su" ] && [ "$su" != "$(id -un)" ] && AS="sudo -u $su"
  local k t h p
  while IFS="$TAB" read -r k t h p <&3; do
    case "$k" in
      envfile) [ -f "$p" ] && [ "$h" != - ] && [ "$(sha_of "$p")" = "$h" ] && xr $AS rm -f "$p" ;;
      model|modelsums) [ -n "$PURGE_MODELS" ] && xr $AS rm -f "$p" ;;
    esac
  done 3< "$M"
  xr $AS rm -rf "$SROOT/server" "$SROOT/venv" "$SROOT/canary" "$SROOT/VERSION" "$M"
  for p in "$SROOT"/models/* "$SROOT/models" "$SROOT/logs" "$SROOT/backups" "$SROOT"; do
    [ -d "$p" ] && xr $AS rmdir "$p" 2>/dev/null
  done
  echo "cwtools-infer uninstalled${DRY:+ (dry-run)}.$([ -d "$SROOT" ] && echo " Left in $SROOT: $(ls -A "$SROOT" | tr '\n' ' ')")"
}

# ═══════════════════════════════════════════════════════════ main ══
# options are checked before any question is asked
[ -n "$OPT_MODELS" ] && { opt_check --models "$OPT_MODELS" chk_models_mode; OPT_MODELS="$VALID"; }
[ -n "$OPT_REMOTE" ] && { opt_check --remote-server "$OPT_REMOTE" chk_url; OPT_REMOTE="$VALID"; }
[ -n "$OPT_PORT" ] && opt_check --port "$OPT_PORT" chk_port
[ -n "$OPT_SERVICE" ] && opt_check --service "$OPT_SERVICE" chk_service
[ -n "$OPT_SVC_USER" ] && opt_check --service-user "$OPT_SVC_USER" chk_user
[ -n "$ALLOW_SET" ] && { opt_check --allow "$OPT_ALLOW" chk_allow; OPT_ALLOW="$VALID"; }
for x in $(lc "$OPT_MODEL_SETS" | tr ',' ' '); do
  case "$x" in default|all) ;; *) set_exists "$x" || die "--model-sets: unknown model set $x (see --list-models)" ;; esac
done
if [ -n "$LIST_MODELS" ]; then list_models "$([ -n "$SERVER" ] && echo server)"; echo "From: $MODELS_URL"; exit 0; fi
if [ -n "$SERVER" ]; then
  case "$MODE" in install) server_install ;; check) server_check ;; uninstall) server_uninstall ;; esac
else
  case "$MODE" in install) plugins_install ;; check) plugins_check ;; uninstall) plugins_uninstall ;; esac
fi
