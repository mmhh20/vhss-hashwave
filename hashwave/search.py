from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

from .expr import Expr, build_from_digest, canonical, crossover, eval_expr, mutate, size, state_hash, to_source
from .miner import MAX_TARGET, WorkHeader, scan_cpu, sha256d, target_from_zero_bits


def _mae(outputs: Sequence[int], targets: Sequence[int]) -> float:
    if len(outputs) != len(targets) or not outputs:
        raise ValueError("outputs and targets must have equal non-zero length")
    return sum(abs(a - b) for a, b in zip(outputs, targets)) / len(targets)


def derive_mutation_seed(
    share_digest: bytes,
    nonce: int,
    parent_state_hash: bytes,
    state_root: bytes,
    generation: int = 0,
) -> bytes:
    """Derive an unconstrained deterministic seed from an accepted PoW share."""
    if len(share_digest) != 32 or len(parent_state_hash) != 32 or len(state_root) != 32:
        raise ValueError("digest and hashes must be 32 bytes")
    if not 0 <= nonce <= 0xFFFFFFFF or not 0 <= generation <= 0xFFFFFFFF:
        raise ValueError("nonce and generation must be uint32")
    return hashlib.sha256(
        b"HashWave/mutation/v3"
        + share_digest
        + nonce.to_bytes(4, "little")
        + parent_state_hash
        + state_root
        + generation.to_bytes(4, "little")
    ).digest()


@dataclass(frozen=True)
class Task:
    name: str
    train_xs: tuple[int, ...]
    train_ys: tuple[int, ...]
    validation_xs: tuple[int, ...] = ()
    validation_ys: tuple[int, ...] = ()
    test_xs: tuple[int, ...] = ()
    test_ys: tuple[int, ...] = ()
    allowed_binary: tuple[str, ...] = ("add", "sub", "mul")
    allowed_unary: tuple[str, ...] = ("neg",)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must not be empty")
        if not self.train_xs or len(self.train_xs) != len(self.train_ys):
            raise ValueError("training split must be non-empty and aligned")
        if len(self.validation_xs) != len(self.validation_ys):
            raise ValueError("validation split is not aligned")
        if len(self.test_xs) != len(self.test_ys):
            raise ValueError("test split is not aligned")
        from .expr import BINARY, UNARY
        if not self.allowed_binary or any(op not in BINARY for op in self.allowed_binary):
            raise ValueError("allowed_binary must be a non-empty subset of supported operators")
        if any(op not in UNARY for op in self.allowed_unary):
            raise ValueError("allowed_unary contains unsupported operators")

    @classmethod
    def from_function(
        cls,
        name: str,
        function: Callable[[int], int],
        *,
        train_xs: Sequence[int],
        validation_xs: Sequence[int] = (),
        test_xs: Sequence[int] = (),
        allowed_binary: tuple[str, ...] = ("add", "sub", "mul"),
        allowed_unary: tuple[str, ...] = ("neg",),
    ) -> "Task":
        train = tuple(int(x) for x in train_xs)
        validation = tuple(int(x) for x in validation_xs)
        test = tuple(int(x) for x in test_xs)
        return cls(
            name=name,
            train_xs=train,
            train_ys=tuple(int(function(x)) for x in train),
            validation_xs=validation,
            validation_ys=tuple(int(function(x)) for x in validation),
            test_xs=test,
            test_ys=tuple(int(function(x)) for x in test),
            allowed_binary=allowed_binary,
            allowed_unary=allowed_unary,
        )

    @classmethod
    def polynomial_demo(cls) -> "Task":
        return cls.from_function(
            "3*x^2+2*x+1",
            lambda x: 3 * x * x + 2 * x + 1,
            train_xs=range(-6, 7),
            validation_xs=(-10, -8, 8, 10),
            test_xs=(-20, -15, 15, 20),
        )

    @property
    def fingerprint(self) -> bytes:
        payload = json.dumps(
            {
                "name": self.name,
                "train_xs": self.train_xs,
                "train_ys": self.train_ys,
                "validation_xs": self.validation_xs,
                "validation_ys": self.validation_ys,
                "allowed_binary": self.allowed_binary,
                "allowed_unary": self.allowed_unary,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(b"HashWave/task/v1" + payload).digest()


@dataclass
class Candidate:
    expr: Expr
    train_mae: float
    validation_mae: float
    train_outputs: tuple[int, ...]
    validation_outputs: tuple[int, ...]
    state_hash: bytes
    semantic_cell: bytes
    generation: int = 0
    parent_hash: bytes = b"\x00" * 32
    support: int = 1
    novelty: float = 0.0

    @property
    def source(self) -> str:
        return to_source(self.expr)

    @property
    def exact_train(self) -> bool:
        return self.train_mae == 0.0

    @property
    def exact_validation(self) -> bool:
        return self.validation_mae == 0.0


@dataclass(frozen=True)
class SearchConfig:
    generations: int = 80
    beam_width: int = 32
    candidates_per_parent: int = 96
    qualifier_bits: int = 0
    max_hash_attempt_multiplier: int = 64
    max_cpu_hash_attempts: int = 1_000_000
    max_nodes: int = 23
    max_depth: int = 9
    complexity_penalty: float = 0.001
    validation_weight: float = 0.35
    novelty_weight: float = 0.02
    semantic_support_weight: float = 0.0
    crossover_rate: float = 0.20
    elite_fraction: float = 0.20
    patience: int = 25
    seed: str = "hashwave-v1"
    source: Literal["hash", "prng"] = "hash"
    chain_bits: int = 0x207FFFFF

    def __post_init__(self) -> None:
        if self.generations <= 0 or self.beam_width <= 0 or self.candidates_per_parent <= 0:
            raise ValueError("search counts must be positive")
        if not 0 <= self.qualifier_bits <= 24:
            raise ValueError("local qualifier_bits must be in [0, 24]")
        if self.max_hash_attempt_multiplier < 1 or self.max_cpu_hash_attempts < 1:
            raise ValueError("hash attempt limits must be positive")
        if self.max_nodes < 1 or self.max_depth < 1:
            raise ValueError("expression limits must be positive")
        if self.complexity_penalty < 0 or self.validation_weight < 0 or self.novelty_weight < 0:
            raise ValueError("weights must be non-negative")
        if not 0 <= self.crossover_rate <= 1 or not 0 < self.elite_fraction <= 1:
            raise ValueError("invalid rate")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.source not in {"hash", "prng"}:
            raise ValueError("source must be hash or prng")


@dataclass
class SearchResult:
    best: Candidate
    test_mae: float | None
    proposals: int
    unique_evaluations: int
    cache_hits: int
    hash_attempts: int
    elapsed_seconds: float
    stopped_reason: str
    history: list[dict]


@dataclass(frozen=True)
class _Evaluation:
    expr: Expr
    train_mae: float
    validation_mae: float
    train_outputs: tuple[int, ...]
    validation_outputs: tuple[int, ...]
    state_hash: bytes
    semantic_cell: bytes


class HashEvolutionSearch:
    """Deterministic hash-guided symbolic search with semantic diversity.

    SHA-256d supplies deterministic proposal entropy and an optional PoW gate.
    Intelligence comes from local variation, evaluation, memory, and selection;
    the hash itself is not treated as a semantic oracle.
    """

    def __init__(self, task: Task, config: SearchConfig | None = None):
        self.task = task
        self.config = config or SearchConfig()
        self.proposals = 0
        self.unique_evaluations = 0
        self.cache_hits = 0
        self.hash_attempts = 0
        self._evaluation_cache: dict[bytes, _Evaluation] = {}
        self._semantic_visits: dict[bytes, int] = {}
        self._prng = random.Random(self.config.seed)

    def run(self) -> SearchResult:
        started = time.perf_counter()
        initial = [("x",), ("c", 0), ("c", 1), ("c", -1)]
        if "mul" in self.task.allowed_binary:
            initial.append(("mul", ("x",), ("x",)))
        if "add" in self.task.allowed_binary:
            initial.append(("add", ("x",), ("c", 1)))
        if "abs" in self.task.allowed_unary:
            initial.append(("abs", ("x",)))
        beam = [self._candidate(expr, 0, b"\x00" * 32) for expr in initial]
        beam = self._select(beam)
        global_best = min(beam, key=self._quality_key)
        history: list[dict] = []
        stale = 0
        stopped_reason = "generation_limit"

        for generation in range(1, self.config.generations + 1):
            expanded: list[Candidate] = []
            cell_parents: dict[bytes, set[bytes]] = {}
            elites = max(1, round(self.config.beam_width * self.config.elite_fraction))
            expanded.extend(beam[:elites])

            for parent_index, parent in enumerate(beam):
                seeds, attempts, state_root = self._proposal_seeds(parent, parent_index, generation)
                self.hash_attempts += attempts
                for proposal_seed in seeds:
                    self.proposals += 1
                    donor = beam[int.from_bytes(proposal_seed[8:12], "little") % len(beam)]
                    base = parent.expr
                    threshold = int(self.config.crossover_rate * 256)
                    if proposal_seed[0] < threshold and donor.state_hash != parent.state_hash:
                        base = crossover(
                            parent.expr,
                            donor.expr,
                            proposal_seed,
                            max_nodes=self.config.max_nodes,
                            max_depth=self.config.max_depth,
                        )
                    child_expr = mutate(
                        base,
                        hashlib.sha256(b"HashWave/local/v1" + proposal_seed).digest(),
                        max_nodes=self.config.max_nodes,
                        max_depth=self.config.max_depth,
                        binary_ops=self.task.allowed_binary,
                        unary_ops=self.task.allowed_unary,
                    )
                    child = self._candidate(child_expr, generation, parent.state_hash)
                    expanded.append(child)
                    cell_parents.setdefault(child.semantic_cell, set()).add(parent.state_hash)

            if not expanded:
                raise RuntimeError("no candidates generated")

            for candidate in expanded:
                candidate.support = len(cell_parents.get(candidate.semantic_cell, {candidate.parent_hash}))
                visits = self._semantic_visits.get(candidate.semantic_cell, 0)
                candidate.novelty = 1.0 / math.sqrt(1.0 + visits)
            for cell in {candidate.semantic_cell for candidate in expanded}:
                self._semantic_visits[cell] = self._semantic_visits.get(cell, 0) + 1

            beam = self._select(expanded)
            current_best = min(beam, key=self._quality_key)
            if self._quality_key(current_best) < self._quality_key(global_best):
                global_best = current_best
                stale = 0
            else:
                stale += 1

            history.append(
                {
                    "generation": generation,
                    "best_train_mae": global_best.train_mae,
                    "best_validation_mae": global_best.validation_mae,
                    "best_program": global_best.source,
                    "best_size": size(global_best.expr),
                    "semantic_cells": len({c.semantic_cell for c in beam}),
                    "proposals": self.proposals,
                    "unique_evaluations": self.unique_evaluations,
                    "cache_hits": self.cache_hits,
                    "hash_attempts": self.hash_attempts,
                }
            )

            if global_best.exact_train and (not self.task.validation_xs or global_best.exact_validation):
                stopped_reason = "exact_train_and_validation"
                break
            if stale >= self.config.patience:
                stopped_reason = "patience"
                break

        test_mae = self._test_mae(global_best.expr)
        return SearchResult(
            best=global_best,
            test_mae=test_mae,
            proposals=self.proposals,
            unique_evaluations=self.unique_evaluations,
            cache_hits=self.cache_hits,
            hash_attempts=self.hash_attempts,
            elapsed_seconds=time.perf_counter() - started,
            stopped_reason=stopped_reason,
            history=history,
        )

    def _proposal_seeds(self, parent: Candidate, parent_index: int, generation: int) -> tuple[list[bytes], int, bytes]:
        state_root = sha256d(
            b"HashWave/state-root/v1"
            + hashlib.sha256(self.config.seed.encode("utf-8")).digest()
            + self.task.fingerprint
            + parent.state_hash
            + generation.to_bytes(4, "little")
        )
        if self.config.source == "prng":
            seeds = [self._prng.randbytes(32) for _ in range(self.config.candidates_per_parent)]
            return seeds, self.config.candidates_per_parent, state_root

        header = WorkHeader(
            version=1,
            previous=parent.state_hash,
            state_root=state_root,
            timestamp=generation,
            bits=self.config.chain_bits,
        )
        target = target_from_zero_bits(self.config.qualifier_bits)
        desired = self.config.candidates_per_parent
        seeds: list[bytes] = []
        attempts = 0
        nonce = (parent_index * 1_000_003 + generation * 10_000_019) & 0xFFFFFFFF
        max_attempts = min(
            self.config.max_cpu_hash_attempts,
            desired * (2**self.config.qualifier_bits) * self.config.max_hash_attempt_multiplier,
        )
        batch = max(64, min(65536, desired * max(1, 2**self.config.qualifier_bits)))

        while len(seeds) < desired and attempts < max_attempts:
            remaining_nonce_space = 0x100000000 - nonce
            requested = (desired - len(seeds)) if self.config.qualifier_bits == 0 else batch
            current = min(requested, remaining_nonce_space, max_attempts - attempts)
            if current <= 0:
                break
            # Consume the whole scanned range so hash_attempts and nonce
            # accounting exactly match the work actually performed.
            batch_shares = list(scan_cpu(header, nonce, current, target=target))
            attempts += current
            nonce += current
            for share in batch_shares:
                seeds.append(
                    derive_mutation_seed(share.digest, share.nonce, parent.state_hash, state_root, generation)
                )
                if len(seeds) >= desired:
                    break
            if nonce > 0xFFFFFFFF:
                break

        if len(seeds) < desired:
            raise RuntimeError(
                f"only {len(seeds)}/{desired} shares found after {attempts} hashes; "
                "lower qualifier_bits or allocate external miner work"
            )
        return seeds, attempts, state_root

    def _candidate(self, expr: Expr, generation: int, parent_hash: bytes) -> Candidate:
        expression = canonical(expr)
        digest = state_hash(expression)
        cached = self._evaluation_cache.get(digest)
        if cached is None:
            train_outputs = tuple(eval_expr(expression, x) for x in self.task.train_xs)
            validation_outputs = tuple(eval_expr(expression, x) for x in self.task.validation_xs)
            train_mae = _mae(train_outputs, self.task.train_ys)
            validation_mae = _mae(validation_outputs, self.task.validation_ys) if self.task.validation_xs else 0.0
            semantic_payload = json.dumps(
                {"train": train_outputs, "validation": validation_outputs}, separators=(",", ":")
            ).encode("utf-8")
            cached = _Evaluation(
                expr=expression,
                train_mae=train_mae,
                validation_mae=validation_mae,
                train_outputs=train_outputs,
                validation_outputs=validation_outputs,
                state_hash=digest,
                semantic_cell=hashlib.sha256(b"HashWave/semantic/v1" + semantic_payload).digest()[:12],
            )
            self._evaluation_cache[digest] = cached
            self.unique_evaluations += 1
        else:
            self.cache_hits += 1
        return Candidate(
            expr=cached.expr,
            train_mae=cached.train_mae,
            validation_mae=cached.validation_mae,
            train_outputs=cached.train_outputs,
            validation_outputs=cached.validation_outputs,
            state_hash=cached.state_hash,
            semantic_cell=cached.semantic_cell,
            generation=generation,
            parent_hash=parent_hash,
        )

    def _objective(self, candidate: Candidate) -> float:
        support_bonus = math.log1p(max(0, candidate.support - 1))
        return (
            candidate.train_mae
            + self.config.validation_weight * candidate.validation_mae
            + self.config.complexity_penalty * size(candidate.expr)
            - self.config.novelty_weight * candidate.novelty
            - self.config.semantic_support_weight * support_bonus
        )

    def _quality_key(self, candidate: Candidate) -> tuple[float, float, float, int, bytes]:
        return (
            candidate.train_mae + self.config.validation_weight * candidate.validation_mae,
            candidate.validation_mae,
            candidate.train_mae,
            size(candidate.expr),
            candidate.state_hash,
        )

    def _select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        # First deduplicate exact syntax, then keep the best/smallest program in
        # each observed semantic cell. This prevents repeated paths from wasting
        # the beam while retaining behavioural diversity.
        by_state: dict[bytes, Candidate] = {}
        for candidate in candidates:
            previous = by_state.get(candidate.state_hash)
            if previous is None or self._objective(candidate) < self._objective(previous):
                by_state[candidate.state_hash] = candidate

        by_cell: dict[bytes, Candidate] = {}
        for candidate in by_state.values():
            previous = by_cell.get(candidate.semantic_cell)
            if previous is None or self._objective(candidate) < self._objective(previous):
                by_cell[candidate.semantic_cell] = candidate

        ordered = sorted(
            by_cell.values(),
            key=lambda c: (self._objective(c), self._quality_key(c)),
        )
        if not ordered:
            raise RuntimeError("selection removed every candidate")
        return ordered[: self.config.beam_width]

    def _test_mae(self, expr: Expr) -> float | None:
        if not self.task.test_xs:
            return None
        outputs = tuple(eval_expr(expr, x) for x in self.task.test_xs)
        return _mae(outputs, self.task.test_ys)


class RandomProgramSearch:
    """Full random-program baseline using the same hash-derived grammar."""

    def __init__(self, task: Task, evaluations: int, seed: str = "hashwave-v1", max_nodes: int = 23):
        if evaluations <= 0:
            raise ValueError("evaluations must be positive")
        self.task = task
        self.evaluations = evaluations
        self.seed = seed
        self.max_nodes = max_nodes

    def run(self) -> SearchResult:
        started = time.perf_counter()
        best: Candidate | None = None
        history: list[dict] = []
        seen: set[bytes] = set()
        for nonce in range(self.evaluations):
            digest = sha256d(b"HashWave/random/v1" + self.seed.encode() + nonce.to_bytes(8, "little"))
            expr = canonical(build_from_digest(
                digest,
                max_nodes=self.max_nodes,
                binary_ops=self.task.allowed_binary,
                unary_ops=self.task.allowed_unary,
            ))
            digest_state = state_hash(expr)
            if digest_state in seen:
                continue
            seen.add(digest_state)
            train_outputs = tuple(eval_expr(expr, x) for x in self.task.train_xs)
            validation_outputs = tuple(eval_expr(expr, x) for x in self.task.validation_xs)
            train_mae = _mae(train_outputs, self.task.train_ys)
            validation_mae = _mae(validation_outputs, self.task.validation_ys) if self.task.validation_xs else 0.0
            semantic = hashlib.sha256(repr((train_outputs, validation_outputs)).encode()).digest()[:12]
            candidate = Candidate(
                expr=expr,
                train_mae=train_mae,
                validation_mae=validation_mae,
                train_outputs=train_outputs,
                validation_outputs=validation_outputs,
                state_hash=digest_state,
                semantic_cell=semantic,
            )
            key = (train_mae + 0.35 * validation_mae, size(expr), digest_state)
            if best is None or key < (best.train_mae + 0.35 * best.validation_mae, size(best.expr), best.state_hash):
                best = candidate
                history.append(
                    {
                        "evaluation": nonce + 1,
                        "train_mae": train_mae,
                        "validation_mae": validation_mae,
                        "program": candidate.source,
                    }
                )
            if candidate.exact_train and (not self.task.validation_xs or candidate.exact_validation):
                break
        assert best is not None
        test_outputs = tuple(eval_expr(best.expr, x) for x in self.task.test_xs)
        test_mae = _mae(test_outputs, self.task.test_ys) if self.task.test_xs else None
        return SearchResult(
            best=best,
            test_mae=test_mae,
            proposals=nonce + 1,
            unique_evaluations=len(seen),
            cache_hits=(nonce + 1) - len(seen),
            hash_attempts=nonce + 1,
            elapsed_seconds=time.perf_counter() - started,
            stopped_reason="exact_train_and_validation" if best.exact_train and best.exact_validation else "evaluation_limit",
            history=history,
        )


# Backward-compatible alias for old scripts; the misleading name is deprecated.
HashInterferenceSearch = HashEvolutionSearch
RandomHashSearch = RandomProgramSearch
