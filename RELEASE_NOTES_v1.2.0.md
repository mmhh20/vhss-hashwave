# VHSS v1.2.0 — Public research-software release

## Scope

This release provides the reference implementation and reproducibility assets
for Verifiable Hash-Seeded Symbolic Search (VHSS).

## Included

- deterministic SHA-256d work-header and target validation;
- domain-separated mutation-seed derivation;
- symbolic expression search with local mutation, crossover, caching, and selection;
- PRNG and exploratory algorithmic baselines;
- 30-task SN Computer Science benchmark assets;
- 43 automated tests;
- documentation of the mathematical model, limitations, and future ASIC protocol.

## Scientific boundary

This release does not claim that SHA-256 has intrinsic intelligence or a
statistically established search advantage over conventional PRNG seeding.
No physical SHA-256 ASIC, Stratum server, or energy meter was used. Hardware
performance and energy efficiency remain future validation requirements.

## Release asset

Attach `vhss_hashwave-1.2.0-py3-none-any.whl` to the GitHub Release rather than
committing it to the source tree.
