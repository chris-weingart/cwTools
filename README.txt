cwTools - ML matting nodes for Autodesk Flame
https://github.com/chris-weingart/cwTools

INSTALL

  ./install.sh              shows the plan; Enter installs, a line number changes that line
  ./install.sh --help       every option
  ./install.sh --check      verify an install
  ./install.sh --uninstall  remove it

Then restart Flame. If a new or updated plugin does not show up, delete Flame's
ofxPluginCache.xml (/opt/Autodesk/.flame*/status/) so the plugins are rescanned.

MODELS

The models are separate downloads, checked by sha256. By default the installer
downloads the sets the chosen tools need from the release. Other options:

  ./install.sh --list-models                          sets, sizes, licences
  ./install.sh --models from:/folder                  offline: archives you downloaded
  ./install.sh --models shared:/mnt/share/cwTools     one models folder for the studio

REMOTE SERVER (optional, Linux + NVIDIA GPU)

  ./install.sh --server     runs cwRefine / cwMatte models on a GPU server.
                            No authentication: allow only your LAN or VPN (--allow).

DOCUMENTATION

Per-tool pages, requirements and the Remote server:
https://github.com/chris-weingart/cwTools

LICENCES

cwTools is Apache-2.0 (LICENSE). The models keep their own licences: see
THIRD_PARTY_NOTICES.txt and licenses/. cwSegment's SAM3 models are under Meta's
SAM License; the installer asks you to accept it.
