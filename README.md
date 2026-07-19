# VHSS / HashWave
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21435910.svg)](https://doi.org/10.5281/zenodo.21435910)
[![Tests](https://github.com/mmhh20/vhss-hashwave/actions/workflows/tests.yml/badge.svg)](https://github.com/mmhh20/vhss-hashwave/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

**VHSS (Verifiable Hash-Seeded Symbolic Search)** is an open-source research
software package for studying a narrow question: can a verified SHA-256d
proof-of-work receipt be transformed into a deterministic, unbiased mutation
seed for symbolic program search?

The software does **not** claim that hashes are intelligent, quantum states,
or superior random sources. Its contribution is an auditable interface between
proof-of-work receipts and a symbolic evolutionary search engine. Search quality
comes from local mutation, crossover, evaluation, caching, and selection.

## Research use

VHSS supports reproducible experiments on:

- deterministic symbolic regression;
- hash-seeded versus conventional pseudo-random evolutionary search;
- Bitcoin-style 80-byte work headers and target validation;
- domain-separated seed derivation from accepted work receipts;
- baseline random-program search;
- limited hyperdimensional associative memory experiments;
- design of future, explicitly **unvalidated**, physical-miner experiments.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

No third-party runtime dependency is required.

## Quick start

```bash
vhss demo --generations 20 --beam 16 --per-parent 32 --seed example --json
vhss benchmark --json
vhss memory-demo
```

The legacy `hashwave` command remains available.

## Verification

```bash
python -m unittest discover -s tests -v
python experiments/full_validation.py --seeds 5
PYTHONPATH=.:experiments python experiments/hash_vs_prng.py --seeds 20
```

A clean test run currently contains 43 automated tests. Coverage, packaging,
and reproducibility instructions are in [docs/reproducibility.md](docs/reproducibility.md).

## Core design

For a verified work digest `d`, nonce `n`, parent state hash `h_p`, task state
root `r`, and generation `g`, VHSS derives a mutation seed

```text
u = SHA256(ASCII("HashWave/mutation/v3") || d || LE32(n) || h_p || r || LE32(g))
```

The exact ASCII domain-separation tag used by VHSS/HashWave version 1.2.0 is
`HashWave/mutation/v3`. `LE32` denotes unsigned 32-bit little-endian
serialization. The archived version 1.2.0 implementation uses this byte
layout. This documentation correction changes no source-code logic,
data, tests, figures, or numerical results.

VHSS then applies a bounded deterministic transformation

```text
p_(t+1) = Delta(p_t, u).
```

The proof-of-work receipt proves expenditure under a declared target; it does
not evaluate candidate quality. Candidate quality is measured separately on
training and validation splits. See [docs/mathematical_model.md](docs/mathematical_model.md).

## Important limitation

No physical ASIC miner, Stratum server, energy meter, or multi-node blockchain
was used in the published software validation. Hardware claims are therefore
out of scope for the current release. The protocol and measurements required
for future validation are specified in
[docs/hardware_validation.md](docs/hardware_validation.md).

## Documentation

- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Concepts and architecture](docs/architecture.md)
- [Mathematical model](docs/mathematical_model.md)
- [API reference](docs/api.md)
- [Reproducibility](docs/reproducibility.md)
- [Hardware validation protocol](docs/hardware_validation.md)
- [Limitations](docs/limitations.md)
- [Contributing](CONTRIBUTING.md)
- [Support](SUPPORT.md)

## Citation

Use [CITATION.cff](CITATION.cff) to cite this software.

**Mehdi Moradi**  
Independent Researcher, Dorud, Lorestan, Iran  
ORCID: [0009-0008-2010-2393](https://orcid.org/0009-0008-2010-2393)

Moradi, M. (2026). *VHSS: Verifiable Hash-Seeded Symbolic Search*  
(Version 1.2.0). Zenodo.  
https://doi.org/10.5281/zenodo.21435911

Concept DOI: https://doi.org/10.5281/zenodo.21435910

## License

BSD 3-Clause License. See [LICENSE](LICENSE).


## SN Computer Science benchmark

The expanded evaluation used in the submission manuscript contains 30 tasks, 300 paired SHA-256-versus-PRNG runs, and an exploratory 90-run six-method comparison. Reproduce the results with:

```bash
python experiments/sncs_benchmark.py --seeds 10 --methods vhss_hash,vhss_prng --output results/sncs/primary.json
python experiments/sncs_benchmark.py --seeds 3 --methods vhss_hash,vhss_prng,standard_gp,beam_mutation,simulated_annealing,random_program --output results/sncs/baselines.json
```

The manuscript does not claim an intrinsic search advantage for SHA-256 or validated ASIC energy efficiency.
