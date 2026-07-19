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

## Archival identifiers

- Version DOI: https://doi.org/10.5281/zenodo.21435911
- Project concept DOI: https://doi.org/10.5281/zenodo.21435910

## Documentation correction — 2026-07-19

The README originally displayed the mutation-seed domain-separation tag as
`VHSS/mutation/v1`. The version 1.2.0 implementation in `hashwave/search.py`
uses `HashWave/mutation/v3`, with nonce and generation serialized as unsigned
32-bit little-endian integers. This is a documentation-only correction. No
algorithmic code, datasets, tests, figures, package version, or reported
numerical results changed.

## Release assets

GitHub automatically provides source-code archives for the release. A wheel
named `vhss_hashwave-1.2.0-py3-none-any.whl` may be attached as an optional
installation convenience; it is not required for exact reproduction and must
not be committed to the source tree.
