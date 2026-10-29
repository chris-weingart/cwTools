"""Allow-list tests (stdlib only): python3 -m unittest server/test_netallow.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netallow as na  # noqa: E402


class AllowTest(unittest.TestCase):
    def test_default_is_loopback(self):
        nets = na.parse_allow(na.DEFAULT_ALLOW)
        self.assertTrue(na.loopback_only(nets))
        self.assertTrue(na.is_allowed("127.0.0.1", nets))
        self.assertTrue(na.is_allowed("127.4.5.6", nets))
        self.assertTrue(na.is_allowed("::1", nets))
        self.assertTrue(na.is_allowed("::ffff:127.0.0.1", nets))
        self.assertFalse(na.is_allowed("192.168.1.10", nets))
        self.assertFalse(na.is_allowed("::ffff:192.168.1.10", nets))

    def test_lan_and_single_host(self):
        nets = na.parse_allow(" 10.0.0.0/8 , 192.168.1.20,fd00::/8,")
        self.assertFalse(na.loopback_only(nets))
        self.assertTrue(na.is_allowed("10.1.2.3", nets))
        self.assertTrue(na.is_allowed("192.168.1.20", nets))
        self.assertFalse(na.is_allowed("192.168.1.21", nets))
        self.assertTrue(na.is_allowed("fd12::5", nets))
        self.assertFalse(na.is_allowed("127.0.0.1", nets))

    def test_host_bits_tolerated(self):
        self.assertEqual(str(na.parse_allow("192.168.1.7/24")[0]), "192.168.1.0/24")

    def test_invalid_entries(self):
        for bad in ("10.0.0.0/33", "lan", "300.1.1.1", ",", ""):
            with self.assertRaises(ValueError):
                na.parse_allow(bad)

    def test_garbage_peer(self):
        self.assertFalse(na.is_allowed("not-an-ip", na.parse_allow("0.0.0.0/0")))

    def test_host_loopback(self):
        self.assertTrue(na.host_is_loopback("127.0.0.1"))
        self.assertTrue(na.host_is_loopback("::1"))
        self.assertTrue(na.host_is_loopback("localhost"))
        self.assertFalse(na.host_is_loopback("0.0.0.0"))
        self.assertFalse(na.host_is_loopback("192.168.1.5"))
        self.assertFalse(na.host_is_loopback("render01"))


if __name__ == "__main__":
    unittest.main()
