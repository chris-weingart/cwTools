# Remote server

cwRefine and cwMatte can run their model on a separate Linux GPU machine
instead of the workstation's GPU. This is useful when Flame leaves too
little VRAM for large crops or the 2048 models, or on workstations without a
suitable GPU.

## How it works

The plugin still does all pre- and post-processing (trimap, crop, colour,
resize). It sends only the model input over the network, losslessly
compressed, and gets the model output back. The server runs exactly the same
ONNX files with the same ONNX Runtime version and settings, so Local and
Remote mattes match. The plugin also checks that both sides hold the
identical model file (sha256).

- No authentication. Clients are limited by an allow-list (`CWINFER_ALLOW`,
  networks in CIDR form, loopback only by default). Run it on a trusted
  network or VPN only, never reachable from the internet.
- Integrity checks: model hash at load, finite weights, a reference
  ("canary") inference after every load, NaN/Inf checks on every output.
- GPU etiquette for a shared card: models load on demand and unload after
  an idle time, one inference runs at a time, and memory is returned after
  each run.

## Install

On the GPU machine (Linux, NVIDIA driver, Python 3.10+):

```
./install.sh --server
```

It asks for the install folder, the listen address and port (default
8201) and which clients may connect (`--allow`, e.g.
`127.0.0.0/8,192.168.1.0/24`; it offers the local network when the server
listens on it). It sets up a Python environment with ONNX Runtime GPU 1.22,
downloads and checks the models, builds the canaries and registers a systemd
service.
If the GPU machine runs a firewall, open the port (default 8201/tcp) for
the workstations you allowed, e.g. `sudo firewall-cmd --add-port=8201/tcp
--permanent && sudo firewall-cmd --reload` (firewalld) or `sudo ufw allow
from 192.168.1.0/24 to any port 8201 proto tcp` (ufw). The installer doesn't
change firewall settings.

Then point the plugins at it: the installer on each workstation asks for the
server address, or set it in a node's Remote group. On a Mac with a per-app
network filter (Little Snitch, LuLu, a company firewall), allow Flame to
connect to the server's address and port, or Remote renders time out.

Protocol details: [server/API.md](../server/API.md).

cwSegment's Remote mode uses a separate SAM3 server, which isn't part of
this package yet.
