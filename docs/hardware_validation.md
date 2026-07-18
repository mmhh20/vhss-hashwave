# Hardware validation protocol

Physical-miner validation is **required before making any claim about ASIC
throughput, energy efficiency, operational feasibility, or useful-work mining**.
It has not been completed for this release.

## Minimum testbed

- one documented SHA-256 ASIC miner and firmware version;
- isolated private network;
- Stratum-compatible test coordinator;
- calibrated wall-power meter;
- timestamped network capture;
- reference CPU seed generator;
- identical downstream candidate evaluator.

## Required measurements

1. accepted, stale, duplicate, and rejected shares;
2. effective hash rate and assigned share target;
3. end-to-end latency from job assignment to candidate evaluation;
4. wall energy in joules per accepted receipt;
5. candidates evaluated per second and per kilowatt-hour;
6. comparison with CPU/PRNG seed generation at an equal evaluation budget;
7. thermal stability, disconnect behavior, and nonce-range handling;
8. exact byte-order agreement between coordinator and miner.

## Experimental controls

Candidate quality must be evaluated independently from hash rarity. The same
parent populations, evaluator, search budget, and tasks must be used across seed
sources. Raw traces and meter readings should be archived.

## Acceptance criteria

A hardware claim is publishable only if it is reproducible on at least two runs,
reports uncertainty, names all hardware and firmware, and distinguishes security
work from optimization quality. A failed or energetically unfavorable result
must be reported without reinterpretation as an AI speedup.
