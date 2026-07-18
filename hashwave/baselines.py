from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

from .expr import Expr, build_from_digest, canonical, crossover, eval_expr, mutate, size, state_hash, to_source
from .search import Candidate, SearchResult, Task


def _mae(outputs: Sequence[int], targets: Sequence[int]) -> float:
    if len(outputs) != len(targets) or not outputs:
        raise ValueError("outputs and targets must have equal non-zero length")
    return sum(abs(a - b) for a, b in zip(outputs, targets)) / len(targets)


@dataclass(frozen=True)
class BaselineConfig:
    evaluations: int = 12000
    population_size: int = 64
    tournament_size: int = 4
    crossover_rate: float = 0.70
    elite_fraction: float = 0.10
    max_nodes: int = 25
    max_depth: int = 10
    validation_weight: float = 0.35
    complexity_penalty: float = 0.001
    seed: str = "vhss-baseline"

    def __post_init__(self) -> None:
        if self.evaluations < 8:
            raise ValueError("evaluations must be at least 8")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid tournament size")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("invalid crossover rate")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("invalid elite fraction")


class _Evaluator:
    def __init__(self, task: Task, validation_weight: float, complexity_penalty: float):
        self.task = task
        self.validation_weight = validation_weight
        self.complexity_penalty = complexity_penalty
        self.cache: dict[bytes, Candidate] = {}
        self.cache_hits = 0
        self.unique_evaluations = 0

    def evaluate(self, expr: Expr, generation: int = 0, parent_hash: bytes = b"\x00" * 32) -> Candidate:
        expr = canonical(expr)
        key = state_hash(expr)
        cached = self.cache.get(key)
        if cached is not None:
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
        train_outputs = tuple(eval_expr(expr, x) for x in self.task.train_xs)
        validation_outputs = tuple(eval_expr(expr, x) for x in self.task.validation_xs)
        train_mae = _mae(train_outputs, self.task.train_ys)
        validation_mae = _mae(validation_outputs, self.task.validation_ys) if self.task.validation_xs else 0.0
        semantic = hashlib.sha256(repr((train_outputs, validation_outputs)).encode("utf-8")).digest()[:12]
        candidate = Candidate(
            expr=expr,
            train_mae=train_mae,
            validation_mae=validation_mae,
            train_outputs=train_outputs,
            validation_outputs=validation_outputs,
            state_hash=key,
            semantic_cell=semantic,
            generation=generation,
            parent_hash=parent_hash,
        )
        self.cache[key] = candidate
        self.unique_evaluations += 1
        return candidate

    def objective(self, c: Candidate) -> float:
        return c.train_mae + self.validation_weight * c.validation_mae + self.complexity_penalty * size(c.expr)

    def key(self, c: Candidate) -> tuple:
        return (self.objective(c), c.validation_mae, c.train_mae, size(c.expr), c.state_hash)

    def test_mae(self, expr: Expr) -> float | None:
        if not self.task.test_xs:
            return None
        return _mae(tuple(eval_expr(expr, x) for x in self.task.test_xs), self.task.test_ys)


def _seed_bytes(rng: random.Random) -> bytes:
    return rng.getrandbits(256).to_bytes(32, "big")


def _initial_population(task: Task, evaluator: _Evaluator, rng: random.Random, count: int, max_nodes: int, max_depth: int) -> list[Candidate]:
    atoms: list[Expr] = [("x",), ("c", 0), ("c", 1), ("c", -1)]
    population = [evaluator.evaluate(x) for x in atoms]
    while len(population) < count:
        expr = build_from_digest(
            _seed_bytes(rng), max_nodes=max_nodes, max_depth=max_depth,
            binary_ops=task.allowed_binary, unary_ops=task.allowed_unary,
        )
        population.append(evaluator.evaluate(expr))
    dedup = {c.state_hash: c for c in population}
    return sorted(dedup.values(), key=evaluator.key)[:count]


class StandardGeneticProgramming:
    """Conventional tournament-selection GP baseline with PRNG variation."""

    def __init__(self, task: Task, config: BaselineConfig | None = None):
        self.task = task
        self.config = config or BaselineConfig()

    def run(self) -> SearchResult:
        started = time.perf_counter()
        rng = random.Random(self.config.seed)
        evaluator = _Evaluator(self.task, self.config.validation_weight, self.config.complexity_penalty)
        population = _initial_population(
            self.task, evaluator, rng, self.config.population_size,
            self.config.max_nodes, self.config.max_depth,
        )
        proposals = len(population)
        best = min(population, key=evaluator.key)
        history = [{"evaluation": proposals, "train_mae": best.train_mae, "validation_mae": best.validation_mae, "program": best.source}]
        generation = 0
        reason = "evaluation_limit"

        def tournament() -> Candidate:
            choices = [population[rng.randrange(len(population))] for _ in range(self.config.tournament_size)]
            return min(choices, key=evaluator.key)

        while proposals < self.config.evaluations:
            generation += 1
            elite_n = max(1, round(self.config.elite_fraction * self.config.population_size))
            next_pop = sorted(population, key=evaluator.key)[:elite_n]
            while len(next_pop) < self.config.population_size and proposals < self.config.evaluations:
                parent = tournament()
                digest = _seed_bytes(rng)
                if rng.random() < self.config.crossover_rate:
                    donor = tournament()
                    expr = crossover(parent.expr, donor.expr, digest, self.config.max_nodes, self.config.max_depth)
                    expr = mutate(
                        expr, hashlib.sha256(b"gp-mutation" + digest).digest(),
                        self.config.max_nodes, self.config.max_depth,
                        binary_ops=self.task.allowed_binary, unary_ops=self.task.allowed_unary,
                    )
                else:
                    expr = mutate(
                        parent.expr, digest, self.config.max_nodes, self.config.max_depth,
                        binary_ops=self.task.allowed_binary, unary_ops=self.task.allowed_unary,
                    )
                next_pop.append(evaluator.evaluate(expr, generation, parent.state_hash))
                proposals += 1
            population = sorted({c.state_hash: c for c in next_pop}.values(), key=evaluator.key)
            while len(population) < self.config.population_size and proposals < self.config.evaluations:
                expr = build_from_digest(
                    _seed_bytes(rng), self.config.max_nodes, self.config.max_depth,
                    binary_ops=self.task.allowed_binary, unary_ops=self.task.allowed_unary,
                )
                population.append(evaluator.evaluate(expr, generation))
                proposals += 1
            current = min(population, key=evaluator.key)
            if evaluator.key(current) < evaluator.key(best):
                best = current
                history.append({"evaluation": proposals, "train_mae": best.train_mae, "validation_mae": best.validation_mae, "program": best.source})
            if best.exact_train and (not self.task.validation_xs or best.exact_validation):
                reason = "exact_train_and_validation"
                break
        return SearchResult(best, evaluator.test_mae(best.expr), proposals, evaluator.unique_evaluations, evaluator.cache_hits, 0, time.perf_counter()-started, reason, history)


class BeamMutationSearch:
    """PRNG local-mutation beam search without crossover or tournament selection."""

    def __init__(self, task: Task, config: BaselineConfig | None = None, beam_width: int = 32):
        self.task = task
        self.config = config or BaselineConfig()
        self.beam_width = beam_width

    def run(self) -> SearchResult:
        started = time.perf_counter()
        rng = random.Random(self.config.seed)
        evaluator = _Evaluator(self.task, self.config.validation_weight, self.config.complexity_penalty)
        beam = _initial_population(self.task, evaluator, rng, self.beam_width, self.config.max_nodes, self.config.max_depth)
        proposals = len(beam)
        best = min(beam, key=evaluator.key)
        history = [{"evaluation": proposals, "train_mae": best.train_mae, "validation_mae": best.validation_mae, "program": best.source}]
        generation = 0
        reason = "evaluation_limit"
        while proposals < self.config.evaluations:
            generation += 1
            children = list(beam[: max(1, self.beam_width // 5)])
            per_parent = max(1, (self.config.evaluations - proposals) // max(1, len(beam)))
            per_parent = min(24, per_parent)
            for parent in beam:
                for _ in range(per_parent):
                    if proposals >= self.config.evaluations:
                        break
                    expr = mutate(
                        parent.expr, _seed_bytes(rng), self.config.max_nodes, self.config.max_depth,
                        binary_ops=self.task.allowed_binary, unary_ops=self.task.allowed_unary,
                    )
                    children.append(evaluator.evaluate(expr, generation, parent.state_hash))
                    proposals += 1
            by_semantic: dict[bytes, Candidate] = {}
            for c in children:
                old = by_semantic.get(c.semantic_cell)
                if old is None or evaluator.key(c) < evaluator.key(old):
                    by_semantic[c.semantic_cell] = c
            beam = sorted(by_semantic.values(), key=evaluator.key)[: self.beam_width]
            current = min(beam, key=evaluator.key)
            if evaluator.key(current) < evaluator.key(best):
                best = current
                history.append({"evaluation": proposals, "train_mae": best.train_mae, "validation_mae": best.validation_mae, "program": best.source})
            if best.exact_train and (not self.task.validation_xs or best.exact_validation):
                reason = "exact_train_and_validation"
                break
        return SearchResult(best, evaluator.test_mae(best.expr), proposals, evaluator.unique_evaluations, evaluator.cache_hits, 0, time.perf_counter()-started, reason, history)


class SimulatedAnnealingSearch:
    """Single-trajectory mutation search with exponential cooling."""

    def __init__(self, task: Task, config: BaselineConfig | None = None, initial_temperature: float = 5.0, cooling: float = 0.999):
        self.task = task
        self.config = config or BaselineConfig()
        self.initial_temperature = initial_temperature
        self.cooling = cooling

    def run(self) -> SearchResult:
        started = time.perf_counter()
        rng = random.Random(self.config.seed)
        evaluator = _Evaluator(self.task, self.config.validation_weight, self.config.complexity_penalty)
        current = evaluator.evaluate(("x",))
        best = current
        proposals = 1
        history = [{"evaluation": 1, "train_mae": best.train_mae, "validation_mae": best.validation_mae, "program": best.source}]
        temperature = self.initial_temperature
        reason = "evaluation_limit"
        while proposals < self.config.evaluations:
            expr = mutate(
                current.expr, _seed_bytes(rng), self.config.max_nodes, self.config.max_depth,
                binary_ops=self.task.allowed_binary, unary_ops=self.task.allowed_unary,
            )
            child = evaluator.evaluate(expr, proposals, current.state_hash)
            proposals += 1
            delta = evaluator.objective(child) - evaluator.objective(current)
            if delta <= 0 or rng.random() < math.exp(-min(700.0, delta / max(temperature, 1e-12))):
                current = child
            if evaluator.key(child) < evaluator.key(best):
                best = child
                history.append({"evaluation": proposals, "train_mae": best.train_mae, "validation_mae": best.validation_mae, "program": best.source})
            if best.exact_train and (not self.task.validation_xs or best.exact_validation):
                reason = "exact_train_and_validation"
                break
            temperature *= self.cooling
        return SearchResult(best, evaluator.test_mae(best.expr), proposals, evaluator.unique_evaluations, evaluator.cache_hits, 0, time.perf_counter()-started, reason, history)
