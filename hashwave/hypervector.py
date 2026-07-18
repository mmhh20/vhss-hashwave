from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HyperVector:
    bits: int
    dimension: int = 4096

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.bits < 0 or self.bits.bit_length() > self.dimension:
            raise ValueError("bits do not fit dimension")

    @classmethod
    def from_label(cls, label: bytes | str, dimension: int = 4096) -> "HyperVector":
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if isinstance(label, str):
            label = label.encode("utf-8")
        chunks: list[bytes] = []
        counter = 0
        while len(chunks) * 256 < dimension:
            chunks.append(hashlib.sha256(b"HashWave/HV/v1" + label + counter.to_bytes(4, "big")).digest())
            counter += 1
        raw = b"".join(chunks)[: (dimension + 7) // 8]
        value = int.from_bytes(raw, "big")
        excess = len(raw) * 8 - dimension
        if excess:
            value >>= excess
        return cls(value, dimension)

    def bind(self, other: "HyperVector") -> "HyperVector":
        self._check(other)
        return HyperVector(self.bits ^ other.bits, self.dimension)

    def permute(self, amount: int = 1) -> "HyperVector":
        amount %= self.dimension
        if amount == 0:
            return self
        mask = (1 << self.dimension) - 1
        value = ((self.bits << amount) | (self.bits >> (self.dimension - amount))) & mask
        return HyperVector(value, self.dimension)

    def similarity(self, other: "HyperVector") -> float:
        self._check(other)
        distance = (self.bits ^ other.bits).bit_count()
        return 1.0 - distance / self.dimension

    def _check(self, other: "HyperVector") -> None:
        if self.dimension != other.dimension:
            raise ValueError("dimension mismatch")


def bundle(vectors: Iterable[HyperVector]) -> HyperVector:
    vectors = list(vectors)
    if not vectors:
        raise ValueError("need at least one vector")
    dimension = vectors[0].dimension
    if any(v.dimension != dimension for v in vectors):
        raise ValueError("dimension mismatch")

    counts = [0] * dimension
    for vector in vectors:
        value = vector.bits
        for i in range(dimension):
            counts[i] += (value >> i) & 1

    threshold = len(vectors) / 2
    out = 0
    for i, count in enumerate(counts):
        # Deterministic tie-breaking prevents order dependence.
        if count > threshold or (count == threshold and i % 2 == 1):
            out |= 1 << i
    return HyperVector(out, dimension)
