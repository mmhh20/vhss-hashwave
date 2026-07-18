# Changelog

## 1.2.0 - 2026-07-18

- Added a 30-task software benchmark and 300 paired SHA-256-versus-PRNG runs.
- Added transparent genetic-programming, beam-search, simulated-annealing, and random-program baselines.
- Added Wilson intervals, paired sign tests, Wilcoxon tests, and Holm adjustment.
- Added publication figures and raw machine-readable results for the SN Computer Science manuscript.
- Clarified that SHA-256 showed no statistically significant intrinsic search advantage.
- Added an explicit protocol for future physical ASIC, Stratum, and energy validation.
- Increased the automated test suite to 43 tests.

## 1.1.0 - 2026-07-18

- Prepared the project for open research-software review.
- Added an OSI-approved BSD 3-Clause license.
- Added an open-source software paper draft and bibliography.
- Added English installation, design, API, reproducibility, limitations, and hardware-validation documentation.
- Added contribution, support, governance, security, citation, and conduct files.
- Added continuous-integration configuration and packaging metadata.
- Added a `vhss` command while retaining `hashwave` compatibility.
- Added coverage threshold and release verification scripts.
- Preserved the central negative result: SHA-256 seeding did not outperform a conventional pseudo-random source in paired software experiments.

## 1.0.0 - 2026-07-18

- Audited reference implementation with correct Bitcoin-style target comparison.
- Forty automated tests.
- Deterministic hash and pseudo-random search baselines.
- Training, validation, and test splits.
- Hyperdimensional memory demonstration.
