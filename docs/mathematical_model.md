# Mathematical model

Let `p_t` be a symbolic program at generation `t`. A Bitcoin-style 80-byte work
header `B` produces

```text
d = SHA256(SHA256(B)).
```

A work receipt is accepted when the little-endian integer represented by `d`
is less than or equal to the declared target `T`:

```text
int_le(d) <= T.
```

Because accepted hashes are conditioned by `T`, VHSS does not directly decode
mutation choices from `d`. Version 1.2.0 derives

```text
u = SHA256(
    ASCII("HashWave/mutation/v3")
    || d
    || LE32(nonce)
    || H(p_t)
    || root(task)
    || LE32(generation)
),
```

where `ASCII("HashWave/mutation/v3")` is the exact domain-separation tag,
`LE32` is unsigned 32-bit little-endian serialization, and `d`, `H(p_t)`, and
`root(task)` are 32-byte values. The implementation rejects non-32-byte digest
or state-hash inputs and nonce or generation values outside the unsigned
32-bit range. The derived value `u` is not subject to the original target
predicate. The next program is

```text
p_(t+1) = Delta(p_t, u),
```

with bounded size and depth. Search can also form a crossover child

```text
p_child = Crossover(p_a, p_b, u).
```

For targets `y_i` and predictions `p(x_i)`, training loss is mean absolute error

```text
MAE_train(p) = (1/N) * sum_i |p(x_i) - y_i|.
```

Selection uses a composite score

```text
L(p) = MAE_train(p)
     + lambda_c * size(p)
     + lambda_v * MAE_validation(p)
     - lambda_n * novelty(p).
```

The exact implementation is documented in `hashwave/search.py`. Test data are
not used during selection; they are evaluated only for the final candidate.

## Interpretation

- SHA-256d supplies a deterministic, externally verifiable seed receipt.
- The mutation operator supplies locality.
- The evaluator supplies task meaning.
- Selection supplies direction.

No quantum superposition or intrinsic intelligence is attributed to hash values.
