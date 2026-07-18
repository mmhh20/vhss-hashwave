import json
import unittest

from hashwave.bridge import ExternalShare, verify_json_line
from hashwave.miner import (
    MAX_TARGET,
    WorkHeader,
    compact_to_target,
    hash_to_pow_int,
    leading_zero_bits_display,
    meets_target,
    scan_cpu,
    sha256d,
    target_from_zero_bits,
    target_to_compact,
)


class CryptoTests(unittest.TestCase):
    def test_bitcoin_genesis_header_hash(self):
        header = bytes.fromhex(
            "01000000" + "00" * 32
            + "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
            + "29ab5f49" + "ffff001d" + "1dac2b7c"
        )
        digest = sha256d(header)
        self.assertEqual(digest.hex(), "6fe28c0ab6f1b372c1a6a246ae63f74f931e8365e15a089c68d6190000000000")
        self.assertTrue(meets_target(digest, compact_to_target(0x1D00FFFF)))
        self.assertEqual(leading_zero_bits_display(digest), 43)

    def test_compact_known_target(self):
        target = compact_to_target(0x1D00FFFF)
        self.assertEqual(target, 0xFFFF * 2 ** (8 * (0x1D - 3)))
        self.assertEqual(target_to_compact(target), 0x1D00FFFF)

    def test_compact_roundtrips_representable_targets(self):
        for compact in (0x01010000, 0x0300FFFF, 0x1B0404CB, 0x1D00FFFF, 0x207FFFFF):
            target = compact_to_target(compact)
            self.assertEqual(compact_to_target(target_to_compact(target)), target)

    def test_compact_rejects_negative_and_overflow(self):
        with self.assertRaises(ValueError):
            compact_to_target(0x1D80FFFF)
        with self.assertRaises(ValueError):
            compact_to_target(0x2300FFFF)

    def test_synthetic_targets(self):
        self.assertEqual(target_from_zero_bits(0), MAX_TARGET)
        self.assertEqual(target_from_zero_bits(256), 0)
        with self.assertRaises(ValueError):
            target_from_zero_bits(257)

    def test_pow_endianness(self):
        raw = bytes.fromhex("ff" * 31 + "00")
        self.assertEqual(hash_to_pow_int(raw), int.from_bytes(raw, "little"))
        self.assertEqual(leading_zero_bits_display(raw), 8)

    def test_header_length_and_validation(self):
        header = WorkHeader(1, b"a" * 32, b"b" * 32, 1, 0x207FFFFF)
        self.assertEqual(len(header.prefix()), 76)
        self.assertEqual(len(header.serialize(7)), 80)
        with self.assertRaises(ValueError):
            WorkHeader(1, b"a", b"b" * 32, 1, 0)
        with self.assertRaises(ValueError):
            header.serialize(-1)

    def test_scan_cpu_all_target(self):
        header = WorkHeader(1, b"a" * 32, b"b" * 32, 1, 0x207FFFFF)
        shares = list(scan_cpu(header, 5, 4, target=MAX_TARGET))
        self.assertEqual([s.nonce for s in shares], [5, 6, 7, 8])

    def test_scan_cpu_rejects_nonce_wrap(self):
        header = WorkHeader(1, b"a" * 32, b"b" * 32, 1, 0)
        with self.assertRaises(ValueError):
            list(scan_cpu(header, 0xFFFFFFFE, 3))

    def test_external_share(self):
        header = WorkHeader(1, b"a" * 32, b"b" * 32, 1, 0)
        share = next(scan_cpu(header, 9, 1))
        verified = ExternalShare(
            header.prefix().hex(), share.nonce, share.digest.hex(), f"{MAX_TARGET:064x}"
        ).verify()
        self.assertEqual(verified, share.digest)

    def test_external_share_rejects_digest_and_target(self):
        header = WorkHeader(1, b"a" * 32, b"b" * 32, 1, 0)
        share = next(scan_cpu(header, 9, 1))
        with self.assertRaises(ValueError):
            ExternalShare(header.prefix().hex(), share.nonce, "00" * 32).verify()
        with self.assertRaises(ValueError):
            ExternalShare(header.prefix().hex(), share.nonce, share.digest.hex(), "0").verify()

    def test_json_share_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            verify_json_line("{}")

    def test_json_share_verifier(self):
        header = WorkHeader(1, b"a" * 32, b"b" * 32, 1, 0)
        share = next(scan_cpu(header, 11, 1))
        line = json.dumps({
            "header_hex": header.prefix().hex(),
            "nonce": share.nonce,
            "digest_hex": share.digest.hex(),
            "target_hex": f"{MAX_TARGET:064x}",
        })
        result = verify_json_line(line)
        self.assertTrue(result["valid"])
        self.assertEqual(result["display_hash"], share.digest[::-1].hex())


if __name__ == "__main__":
    unittest.main()
