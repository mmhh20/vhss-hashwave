from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterator

UINT32_MAX = (1 << 32) - 1
MAX_TARGET = (1 << 256) - 1


def sha256d(data: bytes) -> bytes:
    """Bitcoin-style double SHA-256, returned in raw digest byte order."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def hash_to_pow_int(digest: bytes) -> int:
    """Interpret a raw SHA-256d digest as Bitcoin's little-endian uint256."""
    if len(digest) != 32:
        raise ValueError("digest must be 32 bytes")
    return int.from_bytes(digest, "little")


def target_from_zero_bits(zero_bits: int) -> int:
    """Return a synthetic target with approximately ``zero_bits`` leading bits.

    This is convenient for local experiments. Production networks should use an
    explicit integer target or compact nBits target, not this helper.
    """
    if not 0 <= zero_bits <= 256:
        raise ValueError("zero_bits must be in [0, 256]")
    if zero_bits == 256:
        return 0
    return (1 << (256 - zero_bits)) - 1


def meets_target(digest: bytes, target: int) -> bool:
    if not 0 <= target <= MAX_TARGET:
        raise ValueError("target must be a uint256")
    return hash_to_pow_int(digest) <= target


def compact_to_target(compact: int) -> int:
    """Decode Bitcoin's unsigned compact target representation.

    Negative compact values are rejected because proof-of-work targets are
    unsigned. Overflow beyond 256 bits is also rejected.
    """
    if not 0 <= compact <= UINT32_MAX:
        raise ValueError("compact must be uint32")
    exponent = compact >> 24
    mantissa = compact & 0x007FFFFF
    negative = bool(compact & 0x00800000)
    if negative:
        raise ValueError("negative compact target")
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    if target > MAX_TARGET:
        raise ValueError("compact target overflows uint256")
    return target


def target_to_compact(target: int) -> int:
    """Encode a non-negative uint256 target using Bitcoin compact format."""
    if not 0 <= target <= MAX_TARGET:
        raise ValueError("target must be a uint256")
    if target == 0:
        return 0
    size = (target.bit_length() + 7) // 8
    if size <= 3:
        compact = target << (8 * (3 - size))
    else:
        compact = target >> (8 * (size - 3))
    if compact & 0x00800000:
        compact >>= 8
        size += 1
    compact &= 0x007FFFFF
    compact |= size << 24
    return compact


@dataclass(frozen=True)
class WorkHeader:
    version: int
    previous: bytes
    state_root: bytes
    timestamp: int
    bits: int

    def __post_init__(self) -> None:
        if not 0 <= self.version <= UINT32_MAX:
            raise ValueError("version must be uint32")
        if len(self.previous) != 32 or len(self.state_root) != 32:
            raise ValueError("previous and state_root must be 32 bytes")
        if not 0 <= self.timestamp <= UINT32_MAX:
            raise ValueError("timestamp must be uint32")
        if not 0 <= self.bits <= UINT32_MAX:
            raise ValueError("bits must be uint32")

    def prefix(self) -> bytes:
        """Return the 76-byte Bitcoin-compatible header prefix."""
        return (
            struct.pack("<I", self.version)
            + self.previous
            + self.state_root
            + struct.pack("<II", self.timestamp, self.bits)
        )

    def serialize(self, nonce: int) -> bytes:
        if not 0 <= nonce <= UINT32_MAX:
            raise ValueError("nonce must be uint32")
        return self.prefix() + struct.pack("<I", nonce)


@dataclass(frozen=True)
class Share:
    nonce: int
    digest: bytes


def scan_cpu(
    header: WorkHeader,
    start_nonce: int,
    count: int,
    *,
    target: int = MAX_TARGET,
) -> Iterator[Share]:
    """Reference CPU nonce scanner using Bitcoin-compatible target comparison.

    It never silently wraps the 32-bit nonce. An external job allocator must
    change extra-nonce/state-root when the nonce range is exhausted.
    """
    if not 0 <= start_nonce <= UINT32_MAX:
        raise ValueError("start_nonce must be uint32")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count and start_nonce + count - 1 > UINT32_MAX:
        raise ValueError("nonce range exceeds uint32; allocate a new work item")
    if not 0 <= target <= MAX_TARGET:
        raise ValueError("target must be uint256")

    prefix = header.prefix()
    for nonce in range(start_nonce, start_nonce + count):
        digest = sha256d(prefix + struct.pack("<I", nonce))
        if meets_target(digest, target):
            yield Share(nonce=nonce, digest=digest)


def leading_zero_bits_display(digest: bytes) -> int:
    """Count leading zero bits in Bitcoin's conventional displayed hash."""
    if len(digest) != 32:
        raise ValueError("digest must be 32 bytes")
    total = 0
    for byte in reversed(digest):
        if byte == 0:
            total += 8
            continue
        total += 8 - byte.bit_length()
        break
    return total
