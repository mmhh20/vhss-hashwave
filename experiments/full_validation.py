from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from hashwave.expr import canonical, eval_expr, mutate, size
from hashwave.hdc import build_demo_classifier
from hashwave.miner import MAX_TARGET, WorkHeader, meets_target, scan_cpu, sha256d, target_from_zero_bits
from hashwave.search import HashEvolutionSearch, RandomProgramSearch, SearchConfig, Task


def task_catalog() -> dict[str, Task]:
    wide = tuple(x for x in range(-30, 31) if x not in range(-8, 9))
    return {
        "linear": Task.from_function(
            "2*x+1", lambda x: 2 * x + 1,
            train_xs=range(-5, 6), validation_xs=(-8, -7, 7, 8), test_xs=wide,
        ),
        "quadratic": Task.from_function(
            "3*x^2+2*x+1", lambda x: 3*x*x + 2*x + 1,
            train_xs=range(-6, 7), validation_xs=(-10, -8, 8, 10), test_xs=wide,
        ),
        "cubic": Task.from_function(
            "x^3-2*x+4", lambda x: x*x*x - 2*x + 4,
            train_xs=range(-5, 6), validation_xs=(-8, -7, 7, 8), test_xs=wide,
        ),
        "absolute": Task.from_function(
            "abs(3*x-2)+1", lambda x: abs(3*x - 2) + 1,
            train_xs=range(-7, 8), validation_xs=(-12, -9, 9, 12), test_xs=wide,
            allowed_binary=("add", "sub", "mul"), allowed_unary=("neg", "abs"),
        ),
        "bitmix": Task.from_function(
            "(x XOR 3)+(x AND 1)", lambda x: (x ^ 3) + (x & 1),
            train_xs=range(0, 16), validation_xs=range(16, 24), test_xs=range(24, 64),
            allowed_binary=("add", "sub", "xor", "and", "or"), allowed_unary=("neg",),
        ),
        "quartic": Task.from_function(
            "2*x^4-x^2+3", lambda x: 2*x*x*x*x - x*x + 3,
            train_xs=range(-4, 5), validation_xs=(-7, -6, 6, 7), test_xs=wide,
        ),
        "clamp_gap": Task.from_function(
            "max(x,2)-min(x,-2)", lambda x: max(x, 2) - min(x, -2),
            train_xs=range(-8, 9), validation_xs=(-12, -10, 10, 12), test_xs=wide,
            allowed_binary=("add", "sub", "min", "max"), allowed_unary=("neg",),
        ),
    }


def result_dict(result) -> dict:
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
        "size": size(result.best.expr),
    }


def run_job(job: tuple[str, int, int, int, int]) -> dict:
    task_name, seed_index, generations, beam, per_parent = job
    task = task_catalog()[task_name]
    seed = f"validation-{task_name}-{seed_index}"
    common = dict(
        generations=generations,
        beam_width=beam,
        candidates_per_parent=per_parent,
        max_nodes=25 if task_name != "quartic" else 29,
        max_depth=10,
        patience=generations,
        seed=seed,
    )
    hash_result = HashEvolutionSearch(task, SearchConfig(**common, source="hash")).run()
    prng_result = HashEvolutionSearch(task, SearchConfig(**common, source="prng")).run()
    budget = max(hash_result.proposals, prng_result.proposals)
    random_result = RandomProgramSearch(task, budget, seed=seed, max_nodes=common["max_nodes"]).run()
    return {
        "task": task_name,
        "seed": seed_index,
        "methods": {
            "hash_evolution": result_dict(hash_result),
            "prng_evolution": result_dict(prng_result),
            "random_program": result_dict(random_result),
        },
    }


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def comparison_key(row: dict) -> tuple:
    test = row["test_mae"] if row["test_mae"] is not None else float("inf")
    return (test, row["validation_mae"], row["train_mae"], row["proposals"], row["size"])


def summarize(runs: list[dict]) -> dict:
    result: dict = {}
    for task in sorted({row["task"] for row in runs}):
        rows = [row for row in runs if row["task"] == task]
        task_result: dict = {}
        for method in ("hash_evolution", "prng_evolution", "random_program"):
            values = [row["methods"][method] for row in rows]
            task_result[method] = {
                "runs": len(values),
                "train_exact": sum(v["train_mae"] == 0 for v in values),
                "validation_exact": sum(v["validation_mae"] == 0 for v in values),
                "test_exact": sum(v["test_mae"] == 0 for v in values),
                "median_train_mae": statistics.median(v["train_mae"] for v in values),
                "median_validation_mae": statistics.median(v["validation_mae"] for v in values),
                "median_test_mae": statistics.median(v["test_mae"] for v in values),
                "median_proposals": statistics.median(v["proposals"] for v in values),
                "median_unique_evaluations": statistics.median(v["unique_evaluations"] for v in values),
                "median_elapsed_seconds": statistics.median(v["elapsed_seconds"] for v in values),
            }
        wins = ties = losses = 0
        for row in rows:
            a = comparison_key(row["methods"]["hash_evolution"])
            b = comparison_key(row["methods"]["prng_evolution"])
            if a < b:
                wins += 1
            elif a > b:
                losses += 1
            else:
                ties += 1
        task_result["hash_vs_prng"] = {
            "hash_wins": wins,
            "ties": ties,
            "hash_losses": losses,
            "two_sided_sign_p": sign_test_two_sided(wins, losses),
        }
        result[task] = task_result

    all_pairs = []
    for row in runs:
        all_pairs.append((comparison_key(row["methods"]["hash_evolution"]), comparison_key(row["methods"]["prng_evolution"])))
    wins = sum(a < b for a, b in all_pairs)
    losses = sum(a > b for a, b in all_pairs)
    ties = len(all_pairs) - wins - losses
    result["overall_hash_vs_prng"] = {
        "hash_wins": wins,
        "ties": ties,
        "hash_losses": losses,
        "two_sided_sign_p": sign_test_two_sided(wins, losses),
    }
    return result


def avalanche(samples: int = 20000) -> dict:
    rng = random.Random(0xA51C)
    distances = []
    for _ in range(samples):
        raw = bytearray(rng.randbytes(80))
        first = sha256d(bytes(raw))
        bit = rng.randrange(640)
        raw[bit // 8] ^= 1 << (bit % 8)
        second = sha256d(bytes(raw))
        distances.append((int.from_bytes(first, "big") ^ int.from_bytes(second, "big")).bit_count())
    return {
        "samples": samples,
        "mean": statistics.mean(distances),
        "stdev": statistics.pstdev(distances),
        "minimum": min(distances),
        "maximum": max(distances),
    }


def target_acceptance(samples: int = 200000) -> dict:
    header = WorkHeader(1, hashlib.sha256(b"p").digest(), hashlib.sha256(b"r").digest(), 1, 0x207FFFFF)
    rows = {}
    for bits in (4, 8, 12):
        target = target_from_zero_bits(bits)
        accepted = sum(1 for _ in scan_cpu(header, 0, samples, target=target))
        expected = samples / (2**bits)
        sigma = math.sqrt(samples * (2**-bits) * (1 - 2**-bits))
        rows[str(bits)] = {
            "samples": samples,
            "accepted": accepted,
            "expected": expected,
            "z_score": (accepted - expected) / sigma,
        }
    return rows


def mutation_distribution(samples: int = 20000) -> dict:
    parent = canonical(("add", ("x",), ("c", 1)))
    counts: dict[str, int] = {}
    unchanged = 0
    for i in range(samples):
        digest = hashlib.sha256(b"mutation" + i.to_bytes(8, "little")).digest()
        child = mutate(parent, digest, max_nodes=23, max_depth=9)
        key = child[0]
        counts[key] = counts.get(key, 0) + 1
        unchanged += child == parent
    return {"samples": samples, "root_operator_counts": counts, "unchanged": unchanged}


def hdc_validation() -> dict:
    model = build_demo_classifier()
    rows = [
        ("token", "ارزی بساز که تعدادش محدود باشد"),
        ("token", "سکه قابل سوزاندن با عرضه نهایی"),
        ("token", "امکان ضرب توکن جدید وجود نداشته باشد"),
        ("token", "رمز ارز با تعداد ثابت"),
        ("token", "توکن بدون افزایش عرضه"),
        ("token", "یک دارایی قابل سوزاندن با سقف عرضه"),
        ("nft", "برای عکس های من مجموعه دیجیتال یکتا بساز"),
        ("nft", "هر تصویر یک دارایی غیر قابل تعویض باشد"),
        ("nft", "قرارداد مالکیت آثار هنری"),
        ("nft", "ان اف تی برای تصاویر"),
        ("nft", "مجموعه دارایی های یکتا"),
        ("nft", "مالکیت دیجیتال هر اثر جدا باشد"),
        ("voting", "کاربران بتوانند به پیشنهادها رأی بدهند"),
        ("voting", "یک سیستم نظرسنجی روی زنجیره"),
        ("voting", "تصمیم اعضا با شمارش آرا مشخص شود"),
        ("voting", "هر کاربر حق رای داشته باشد"),
        ("voting", "قرارداد رای گیری جمعی"),
        ("voting", "اعضا پیشنهاد را تایید یا رد کنند"),
    ]
    details = []
    for expected, text in rows:
        predicted, scores = model.predict(text)
        ordered = sorted(scores.values(), reverse=True)
        details.append({
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted,
            "margin": ordered[0] - ordered[1],
        })
    return {
        "correct": sum(row["correct"] for row in details),
        "total": len(details),
        "accuracy": sum(row["correct"] for row in details) / len(details),
        "median_margin": statistics.median(row["margin"] for row in details),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--per-parent", type=int, default=64)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--tasks", default=",".join(task_catalog()))
    parser.add_argument("--output", default="results/full_validation.json")
    args = parser.parse_args()

    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    unknown = set(tasks) - set(task_catalog())
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")

    jobs = [(task, seed, args.generations, args.beam, args.per_parent) for task in tasks for seed in range(args.seeds)]
    started = time.perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            runs.append(future.result())
            if index % max(1, len(jobs) // 10) == 0:
                print(f"completed {index}/{len(jobs)}", flush=True)

    output = {
        "configuration": {
            "seeds": args.seeds,
            "generations": args.generations,
            "beam": args.beam,
            "per_parent": args.per_parent,
            "workers": args.workers,
            "tasks": tasks,
        },
        "summary": summarize(runs),
        "avalanche": avalanche(),
        "target_acceptance": target_acceptance(),
        "mutation_distribution": mutation_distribution(),
        "hdc_validation": hdc_validation(),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": sorted(runs, key=lambda row: (row["task"], row["seed"])),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": output["summary"], "elapsed_seconds": output["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
