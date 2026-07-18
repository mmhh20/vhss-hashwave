from __future__ import annotations

import hashlib
import json
from typing import Iterator

Expr = tuple

COMMUTATIVE = {"add", "mul", "xor", "and", "or", "min", "max"}
BINARY = ("add", "sub", "mul", "xor", "and", "or", "min", "max")
UNARY = ("neg", "abs")
CONSTANTS = tuple(range(-8, 9))
LIMIT = 1_000_000_000_000_000_000


def _validate_expr(expr: Expr) -> None:
    if not isinstance(expr, tuple) or not expr:
        raise ValueError("expression must be a non-empty tuple")
    op = expr[0]
    if op == "x":
        if len(expr) != 1:
            raise ValueError("x node has invalid arity")
    elif op == "c":
        if len(expr) != 2 or not isinstance(expr[1], int):
            raise ValueError("constant node has invalid payload")
    elif op in UNARY:
        if len(expr) != 2:
            raise ValueError("unary node has invalid arity")
        _validate_expr(expr[1])
    elif op in BINARY:
        if len(expr) != 3:
            raise ValueError("binary node has invalid arity")
        _validate_expr(expr[1])
        _validate_expr(expr[2])
    else:
        raise ValueError(f"unknown op: {op}")


def atom_from_byte(byte: int) -> Expr:
    if not 0 <= byte <= 255:
        raise ValueError("byte must be in [0, 255]")
    if byte % 5 == 0:
        return ("x",)
    return ("c", CONSTANTS[byte % len(CONSTANTS)])


def eval_expr(expr: Expr, x: int) -> int:
    op = expr[0]
    if op == "x":
        return int(x)
    if op == "c":
        return int(expr[1])
    if op == "neg":
        return _clip(-eval_expr(expr[1], x))
    if op == "abs":
        return abs(eval_expr(expr[1], x))

    a = eval_expr(expr[1], x)
    b = eval_expr(expr[2], x)
    if op == "add":
        return _clip(a + b)
    if op == "sub":
        return _clip(a - b)
    if op == "mul":
        return _clip(a * b)
    if op == "xor":
        return _clip(a ^ b)
    if op == "and":
        return _clip(a & b)
    if op == "or":
        return _clip(a | b)
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    raise ValueError(f"unknown op: {op}")


def _clip(value: int) -> int:
    return max(-LIMIT, min(LIMIT, int(value)))


def size(expr: Expr) -> int:
    op = expr[0]
    if op in {"x", "c"}:
        return 1
    if op in UNARY:
        return 1 + size(expr[1])
    return 1 + size(expr[1]) + size(expr[2])


def depth(expr: Expr) -> int:
    op = expr[0]
    if op in {"x", "c"}:
        return 1
    if op in UNARY:
        return 1 + depth(expr[1])
    return 1 + max(depth(expr[1]), depth(expr[2]))


def to_source(expr: Expr) -> str:
    op = expr[0]
    if op == "x":
        return "x"
    if op == "c":
        return str(expr[1])
    if op == "neg":
        return f"(-{to_source(expr[1])})"
    if op == "abs":
        return f"abs({to_source(expr[1])})"
    if op in {"min", "max"}:
        return f"{op}({to_source(expr[1])}, {to_source(expr[2])})"
    symbols = {"add": "+", "sub": "-", "mul": "*", "xor": "^", "and": "&", "or": "|"}
    return f"({to_source(expr[1])} {symbols[op]} {to_source(expr[2])})"


def serialize(expr: Expr) -> bytes:
    """Stable, language-independent-enough JSON representation for hashing."""
    _validate_expr(expr)
    return json.dumps(expr, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def state_hash(expr: Expr) -> bytes:
    return hashlib.sha256(b"HashWave/expr/v1" + serialize(canonical(expr))).digest()


def canonical(expr: Expr) -> Expr:
    op = expr[0]
    if op == "x":
        return ("x",)
    if op == "c":
        return ("c", _clip(int(expr[1])))
    if op in UNARY:
        if len(expr) != 2:
            raise ValueError("unary node has invalid arity")
        child = canonical(expr[1])
        if op == "neg":
            if child[0] == "c":
                return ("c", _clip(-child[1]))
            if child[0] == "neg":
                return child[1]
        if op == "abs":
            if child[0] == "c":
                return ("c", abs(child[1]))
            if child[0] == "abs":
                return child
        return (op, child)

    if op not in BINARY or len(expr) != 3:
        raise ValueError(f"invalid expression node: {expr!r}")
    left = canonical(expr[1])
    right = canonical(expr[2])
    if op in COMMUTATIVE and serialize(left) > serialize(right):
        left, right = right, left

    if left[0] == "c" and right[0] == "c":
        return ("c", eval_expr((op, left, right), 0))

    zero = ("c", 0)
    one = ("c", 1)
    if op == "add":
        if left == zero:
            return right
        if right == zero:
            return left
    elif op == "sub":
        if right == zero:
            return left
        if left == right:
            return zero
    elif op == "mul":
        if left == zero or right == zero:
            return zero
        if left == one:
            return right
        if right == one:
            return left
    elif op == "xor":
        if left == right:
            return zero
        if left == zero:
            return right
        if right == zero:
            return left
    elif op in {"and", "or", "min", "max"} and left == right:
        return left

    return (op, left, right)


def paths(expr: Expr, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    out = [prefix]
    op = expr[0]
    if op in UNARY:
        out.extend(paths(expr[1], prefix + (1,)))
    elif op in BINARY:
        out.extend(paths(expr[1], prefix + (1,)))
        out.extend(paths(expr[2], prefix + (2,)))
    return out


def get_at(expr: Expr, path: tuple[int, ...]) -> Expr:
    current = expr
    for index in path:
        if index not in (1, 2) or index >= len(current):
            raise ValueError("invalid expression path")
        current = current[index]
    return current


def replace_at(expr: Expr, path: tuple[int, ...], replacement: Expr) -> Expr:
    if not path:
        return replacement
    index = path[0]
    if index not in (1, 2) or index >= len(expr):
        raise ValueError("invalid expression path")
    items = list(expr)
    items[index] = replace_at(items[index], path[1:], replacement)
    return tuple(items)


def _expand_digest(digest: bytes, minimum: int = 64) -> bytes:
    if not digest:
        raise ValueError("digest must not be empty")
    output = bytearray(digest)
    counter = 0
    while len(output) < minimum:
        output.extend(hashlib.sha256(b"HashWave/expand/v1" + digest + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(output)


def build_from_digest(
    digest: bytes,
    max_nodes: int = 9,
    max_depth: int = 6,
    *,
    binary_ops: tuple[str, ...] = BINARY,
    unary_ops: tuple[str, ...] = UNARY,
) -> Expr:
    if max_nodes < 1 or max_depth < 1:
        raise ValueError("limits must be positive")
    if not binary_ops or any(op not in BINARY for op in binary_ops):
        raise ValueError("binary_ops must be a non-empty subset of BINARY")
    if any(op not in UNARY for op in unary_ops):
        raise ValueError("unary_ops must be a subset of UNARY")
    digest = _expand_digest(digest)
    expr: Expr = atom_from_byte(digest[0])
    cursor = 1
    while size(expr) < max_nodes and cursor + 4 < len(digest):
        mode = digest[cursor] % 5
        op = binary_ops[digest[cursor + 1] % len(binary_ops)]
        atom = atom_from_byte(digest[cursor + 2])
        if mode == 0:
            proposed = (op, expr, atom)
        elif mode == 1:
            proposed = (op, atom, expr)
        elif mode == 2:
            proposed = ((unary_ops[digest[cursor + 1] % len(unary_ops)], expr) if unary_ops else (op, expr, atom))
        elif mode == 3:
            # Build a bounded non-recursive two-atom subtree. Recursive digest
            # expansion can otherwise cycle forever when a short suffix keeps
            # selecting the same mode after expansion.
            atom2 = atom_from_byte(digest[cursor + 3])
            sub_op = binary_ops[digest[cursor + 4] % len(binary_ops)]
            proposed = (op, expr, canonical((sub_op, atom, atom2)))
        else:
            proposed = atom
        proposed = canonical(proposed)
        if size(proposed) <= max_nodes and depth(proposed) <= max_depth:
            expr = proposed
        cursor += 3
    return canonical(expr)


def mutate(
    expr: Expr,
    digest: bytes,
    max_nodes: int = 21,
    max_depth: int = 8,
    *,
    binary_ops: tuple[str, ...] = BINARY,
    unary_ops: tuple[str, ...] = UNARY,
) -> Expr:
    if max_nodes < 1 or max_depth < 1:
        raise ValueError("limits must be positive")
    if not binary_ops or any(op not in BINARY for op in binary_ops):
        raise ValueError("binary_ops must be a non-empty subset of BINARY")
    if any(op not in UNARY for op in unary_ops):
        raise ValueError("unary_ops must be a subset of UNARY")
    digest = _expand_digest(digest)
    expr = canonical(expr)
    available_paths = paths(expr)
    path = available_paths[digest[1] % len(available_paths)]
    target = get_at(expr, path)
    mode = digest[0] % 10
    atom = atom_from_byte(digest[2])
    op = binary_ops[digest[3] % len(binary_ops)]

    if mode == 0:
        new = atom
    elif mode == 1:
        new = (op, target, atom)
    elif mode == 2:
        new = (op, atom, target)
    elif mode == 3:
        new = ((unary_ops[digest[3] % len(unary_ops)], target) if unary_ops else (op, target, atom))
    elif mode == 4 and target[0] == "c":
        delta = (digest[4] % 11) - 5
        new = ("c", max(-64, min(64, target[1] + delta)))
    elif mode == 5 and target[0] in BINARY:
        new = (target[0], target[2], target[1])
    elif mode == 6 and target[0] in BINARY:
        new = target[1 + (digest[4] & 1)]
    elif mode == 7:
        new = build_from_digest(digest[4:], max_nodes=min(7, max_nodes), max_depth=min(4, max_depth), binary_ops=binary_ops, unary_ops=unary_ops)
    elif mode == 8 and target[0] in BINARY:
        new_op = binary_ops[digest[4] % len(binary_ops)]
        new = (new_op, target[1], target[2])
    else:
        new = (op, target, build_from_digest(digest[5:], max_nodes=3, max_depth=2, binary_ops=binary_ops, unary_ops=unary_ops))

    candidate = canonical(replace_at(expr, path, new))
    if size(candidate) > max_nodes or depth(candidate) > max_depth:
        return expr
    return candidate


def crossover(left: Expr, right: Expr, digest: bytes, max_nodes: int = 21, max_depth: int = 8) -> Expr:
    digest = _expand_digest(digest)
    left = canonical(left)
    right = canonical(right)
    left_paths = paths(left)
    right_paths = paths(right)
    destination = left_paths[digest[0] % len(left_paths)]
    source = right_paths[digest[1] % len(right_paths)]
    candidate = canonical(replace_at(left, destination, get_at(right, source)))
    if size(candidate) > max_nodes or depth(candidate) > max_depth:
        return left
    return candidate
