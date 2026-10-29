"""Client allow-list for cwtools-infer (CWINFER_ALLOW).

The server has no authentication: the allow-list is the only gate, so it
defaults to loopback. Entries are comma-separated networks in CIDR notation
(IPv4 or IPv6); a bare address means that single host.
"""
from __future__ import annotations

import ipaddress

DEFAULT_ALLOW = "127.0.0.0/8,::1/128"


def parse_allow(spec: str) -> list:
    """'10.0.0.0/8, 192.168.1.20' -> networks. Raises ValueError naming the bad entry."""
    nets = []
    for raw in spec.split(","):
        s = raw.strip()
        if not s:
            continue
        try:
            nets.append(ipaddress.ip_network(s, strict=False))
        except ValueError as exc:
            raise ValueError(f"CWINFER_ALLOW: invalid entry {s!r} ({exc})") from None
    if not nets:
        raise ValueError("CWINFER_ALLOW is empty: no client could connect")
    return nets


def client_ip(addr: str):
    """Peer address -> ip_address, IPv4-mapped IPv6 (::ffff:a.b.c.d) unwrapped."""
    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def is_allowed(addr: str, nets: list) -> bool:
    try:
        ip = client_ip(addr)
    except ValueError:
        return False
    return any(ip.version == n.version and ip in n for n in nets)


def loopback_only(nets: list) -> bool:
    return all(n.is_loopback for n in nets)


def host_is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a host name: assume reachable from the network
