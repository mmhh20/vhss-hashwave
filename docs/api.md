# API reference

## `Task`

Defines training, validation, and test splits and permitted unary/binary
operators. Use `Task.from_function` for synthetic functions or instantiate it
with explicit observations.

## `SearchConfig`

Controls generations, beam width, candidates per parent, expression limits,
loss weights, crossover rate, patience, seed source, and local proof-of-work gate.

## `HashEvolutionSearch`

Runs bounded symbolic evolutionary search. `source="hash"` uses deterministic
SHA-256-derived seeds; `source="prng"` uses a conventional seeded generator while
preserving the same search structure.

## `RandomProgramSearch`

A non-evolutionary baseline that builds independent programs from digest seeds.

## `WorkHeader` and share validation

`hashwave.miner` implements 80-byte serialization, SHA-256d, compact target
conversion, and CPU scanning. `hashwave.bridge` validates external share records.

## Hyperdimensional modules

`hashwave.hypervector` and `hashwave.hdc` provide a limited associative-memory
demonstration. They are not a general natural-language understanding system.
