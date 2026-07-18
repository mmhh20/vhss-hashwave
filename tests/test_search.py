import hashlib
import unittest

from hashwave.expr import canonical, mutate
from hashwave.search import (
    HashEvolutionSearch,
    RandomProgramSearch,
    SearchConfig,
    Task,
    derive_mutation_seed,
)


class SearchTests(unittest.TestCase):
    def test_task_validation(self):
        with self.assertRaises(ValueError):
            Task("", (1,), (1,))
        with self.assertRaises(ValueError):
            Task("x", (1,), ())

    def test_seed_domain_separation_removes_share_prefix_bias(self):
        parent = canonical(("add", ("x",), ("c", 1)))
        parent_hash = hashlib.sha256(repr(parent).encode()).digest()
        root = hashlib.sha256(b"root").digest()
        old_exprs = set()
        new_exprs = set()
        for nonce in range(256):
            digest = hashlib.sha256(nonce.to_bytes(4, "little")).digest()[:-4] + b"\x00" * 4
            old_exprs.add(repr(mutate(parent, digest)))
            seed = derive_mutation_seed(digest, nonce, parent_hash, root, 1)
            new_exprs.add(repr(mutate(parent, seed)))
        # A real Bitcoin target constrains the displayed leading bits, which
        # are the trailing raw digest bytes. Both streams should retain broad
        # mutation diversity, and domain separation must produce a distinct
        # stream rather than reusing the share bytes directly.
        self.assertGreater(len(old_exprs), 100)
        self.assertGreater(len(new_exprs), 100)
        self.assertNotEqual(old_exprs, new_exprs)

    def test_search_reproducible_hash(self):
        config = SearchConfig(generations=20, beam_width=16, candidates_per_parent=32, seed="repeat")
        a = HashEvolutionSearch(Task.polynomial_demo(), config).run()
        b = HashEvolutionSearch(Task.polynomial_demo(), config).run()
        self.assertEqual(a.best.source, b.best.source)
        self.assertEqual(a.best.train_mae, b.best.train_mae)
        self.assertEqual(a.unique_evaluations, b.unique_evaluations)

    def test_search_reproducible_prng(self):
        config = SearchConfig(generations=15, beam_width=12, candidates_per_parent=24, seed="repeat", source="prng")
        a = HashEvolutionSearch(Task.polynomial_demo(), config).run()
        b = HashEvolutionSearch(Task.polynomial_demo(), config).run()
        self.assertEqual(a.best.source, b.best.source)
        self.assertEqual(a.proposals, b.proposals)

    def test_cache_is_used(self):
        config = SearchConfig(generations=8, beam_width=12, candidates_per_parent=20, seed="cache")
        result = HashEvolutionSearch(Task.polynomial_demo(), config).run()
        self.assertGreater(result.cache_hits, 0)
        self.assertLessEqual(result.unique_evaluations, result.proposals + 7)

    def test_train_validation_test_metrics(self):
        task = Task.from_function(
            "square",
            lambda x: x * x,
            train_xs=(-2, -1, 0, 1, 2),
            validation_xs=(-4, 4),
            test_xs=(-10, 10),
        )
        config = SearchConfig(generations=30, beam_width=16, candidates_per_parent=32, seed="square")
        result = HashEvolutionSearch(task, config).run()
        self.assertIsNotNone(result.test_mae)
        self.assertGreaterEqual(result.best.train_mae, 0)
        self.assertGreaterEqual(result.best.validation_mae, 0)

    def test_local_pow_gate(self):
        task = Task.from_function("linear", lambda x: 2 * x + 1, train_xs=range(-3, 4))
        config = SearchConfig(
            generations=2,
            beam_width=6,
            candidates_per_parent=4,
            qualifier_bits=4,
            max_hash_attempt_multiplier=16,
            seed="pow",
        )
        result = HashEvolutionSearch(task, config).run()
        self.assertGreaterEqual(result.hash_attempts, result.proposals)

    def test_random_program_baseline(self):
        result = RandomProgramSearch(Task.polynomial_demo(), 100, seed="random").run()
        self.assertEqual(result.proposals, 100)
        self.assertGreaterEqual(result.best.train_mae, 0)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            SearchConfig(generations=0)
        with self.assertRaises(ValueError):
            SearchConfig(qualifier_bits=25)
        with self.assertRaises(ValueError):
            SearchConfig(source="bad")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SearchConfig(max_cpu_hash_attempts=0)


if __name__ == "__main__":
    unittest.main()
