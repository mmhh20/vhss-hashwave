from __future__ import annotations

import argparse
import json

from . import __version__
from .bridge import verify_json_line
from .hdc import demo_classifier
from .search import HashEvolutionSearch, RandomProgramSearch, SearchConfig, Task


def _result_payload(result) -> dict:
    return {
        "program": result.best.source,
        "train_mae": result.best.train_mae,
        "validation_mae": result.best.validation_mae,
        "test_mae": result.test_mae,
        "proposals": result.proposals,
        "unique_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "hash_attempts": result.hash_attempts,
        "elapsed_seconds": result.elapsed_seconds,
        "stopped_reason": result.stopped_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VHSS deterministic hash-seeded symbolic search")
    parser.add_argument("--version", action="version", version=f"VHSS {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("demo", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--generations", type=int, default=80)
        command.add_argument("--beam", type=int, default=32)
        command.add_argument("--per-parent", type=int, default=96)
        command.add_argument("--qualifier-bits", type=int, default=0)
        command.add_argument("--seed", default="hashwave-v1")
        command.add_argument("--json", action="store_true")

    sub.add_parser("memory-demo")
    verify = sub.add_parser("verify-share")
    verify.add_argument("line", help="JSON object containing header_hex, nonce, optional digest_hex and target_hex")

    args = parser.parse_args()
    if args.command == "memory-demo":
        print(json.dumps(demo_classifier(), ensure_ascii=False, indent=2))
        return
    if args.command == "verify-share":
        print(json.dumps(verify_json_line(args.line), ensure_ascii=False, indent=2))
        return

    task = Task.polynomial_demo()
    config = SearchConfig(
        generations=args.generations,
        beam_width=args.beam,
        candidates_per_parent=args.per_parent,
        qualifier_bits=args.qualifier_bits,
        seed=args.seed,
        source="hash",
    )
    hash_result = HashEvolutionSearch(task, config).run()
    payload = {"task": task.name, "hash_evolution": _result_payload(hash_result)}

    if args.command == "benchmark":
        prng_config = SearchConfig(**{**config.__dict__, "source": "prng"})
        prng_result = HashEvolutionSearch(task, prng_config).run()
        random_result = RandomProgramSearch(task, max(hash_result.proposals, prng_result.proposals), seed=args.seed).run()
        payload["prng_evolution"] = _result_payload(prng_result)
        payload["random_program"] = _result_payload(random_result)

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
