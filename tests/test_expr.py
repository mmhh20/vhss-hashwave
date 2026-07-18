import hashlib
import random
import unittest

from hashwave.expr import (
    BINARY,
    UNARY,
    build_from_digest,
    canonical,
    crossover,
    depth,
    eval_expr,
    mutate,
    serialize,
    size,
    state_hash,
)


def random_expr(rng: random.Random, level: int = 0):
    if level > 4 or rng.random() < 0.3:
        return ("x",) if rng.random() < 0.4 else ("c", rng.randint(-10, 10))
    if rng.random() < 0.2:
        return (rng.choice(UNARY), random_expr(rng, level + 1))
    return (rng.choice(BINARY), random_expr(rng, level + 1), random_expr(rng, level + 1))


class ExprTests(unittest.TestCase):
    def test_basic_evaluation(self):
        expr = canonical(("add", ("mul", ("x",), ("x",)), ("c", 1)))
        self.assertEqual(eval_expr(expr, 3), 10)

    def test_canonical_rules(self):
        self.assertEqual(canonical(("sub", ("x",), ("x",))), ("c", 0))
        self.assertEqual(canonical(("xor", ("x",), ("x",))), ("c", 0))
        self.assertEqual(canonical(("neg", ("neg", ("x",)))), ("x",))
        self.assertEqual(canonical(("abs", ("abs", ("x",)))), ("abs", ("x",)))

    def test_canonical_is_idempotent_and_equivalent(self):
        rng = random.Random(20260718)
        for _ in range(500):
            expr = random_expr(rng)
            canon = canonical(expr)
            self.assertEqual(canonical(canon), canon)
            for x in range(-8, 9):
                self.assertEqual(eval_expr(expr, x), eval_expr(canon, x))

    def test_stable_serialization_hash(self):
        expr = canonical(("add", ("x",), ("c", 1)))
        self.assertEqual(state_hash(expr), state_hash(expr))
        self.assertNotEqual(state_hash(expr), state_hash(("x",)))
        self.assertTrue(serialize(expr).startswith(b"["))

    def test_build_is_deterministic_and_bounded(self):
        digest = hashlib.sha256(b"build").digest()
        a = build_from_digest(digest, max_nodes=13, max_depth=6)
        b = build_from_digest(digest, max_nodes=13, max_depth=6)
        self.assertEqual(a, b)
        self.assertLessEqual(size(a), 13)
        self.assertLessEqual(depth(a), 6)

    def test_mutation_is_deterministic_and_bounded(self):
        parent = ("add", ("x",), ("c", 1))
        for i in range(500):
            digest = hashlib.sha256(i.to_bytes(4, "little")).digest()
            a = mutate(parent, digest, max_nodes=17, max_depth=7)
            b = mutate(parent, digest, max_nodes=17, max_depth=7)
            self.assertEqual(a, b)
            self.assertLessEqual(size(a), 17)
            self.assertLessEqual(depth(a), 7)

    def test_crossover_is_deterministic_and_bounded(self):
        left = ("add", ("x",), ("c", 1))
        right = ("mul", ("x",), ("x",))
        digest = hashlib.sha256(b"cross").digest()
        child = crossover(left, right, digest, max_nodes=9, max_depth=5)
        self.assertEqual(child, crossover(left, right, digest, max_nodes=9, max_depth=5))
        self.assertLessEqual(size(child), 9)
        self.assertLessEqual(depth(child), 5)

    def test_malformed_expression_rejected(self):
        with self.assertRaises(ValueError):
            canonical(("unknown", ("x",), ("c", 1)))
        with self.assertRaises(ValueError):
            canonical(("neg",))

    def test_invalid_limits_and_digest(self):
        with self.assertRaises(ValueError):
            build_from_digest(b"")
        with self.assertRaises(ValueError):
            mutate(("x",), b"")
        with self.assertRaises(ValueError):
            build_from_digest(b"x", max_nodes=0)


if __name__ == "__main__":
    unittest.main()
