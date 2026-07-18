from __future__ import annotations

import json
from dataclasses import dataclass

from .miner import MAX_TARGET, meets_target, sha256d


@dataclass(frozen=True)
class ExternalShare:
    """A 76-byte header prefix plus nonce submitted by an external miner."""

    header_hex: str
    nonce: int
    digest_hex: str | None = None
    target_hex: str | None = None

    def verify(self) -> bytes:
        try:
            raw = bytes.fromhex(self.header_hex)
        except ValueError as exc:
            raise ValueError("header_hex is not valid hexadecimal") from exc
        if len(raw) != 76:
            raise ValueError("header_hex must encode the 76-byte prefix")
        if not 0 <= self.nonce <= 0xFFFFFFFF:
            raise ValueError("nonce must be uint32")
        target = MAX_TARGET
        if self.target_hex is not None:
            try:
                target = int(self.target_hex, 16)
            except ValueError as exc:
                raise ValueError("target_hex is not valid hexadecimal") from exc
            if not 0 <= target <= MAX_TARGET:
                raise ValueError("target_hex must be uint256")

        digest = sha256d(raw + self.nonce.to_bytes(4, "little"))
        if self.digest_hex is not None:
            supplied = self.digest_hex.lower().removeprefix("0x")
            if len(supplied) != 64 or digest.hex() != supplied:
                raise ValueError("digest mismatch")
        if not meets_target(digest, target):
            raise ValueError("share does not satisfy target")
        return digest


def verify_json_line(line: str) -> dict:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("share payload must be an object")
    try:
        header_hex = str(payload["header_hex"])
        nonce = int(payload["nonce"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("payload requires valid header_hex and nonce") from exc
    share = ExternalShare(
        header_hex=header_hex,
        nonce=nonce,
        digest_hex=payload.get("digest_hex"),
        target_hex=payload.get("target_hex"),
    )
    digest = share.verify()
    return {
        "nonce": share.nonce,
        "digest_hex": digest.hex(),
        "display_hash": digest[::-1].hex(),
        "valid": True,
    }
