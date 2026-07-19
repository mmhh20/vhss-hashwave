# Reproducibility

## Automated checks

```bash
coverage erase
coverage run --branch --source=hashwave -m unittest discover -s tests -v
coverage report --fail-under=85
python experiments/full_validation.py --seeds 5
PYTHONPATH=.:experiments python experiments/hash_vs_prng.py --seeds 20
```

## Required metadata for reported experiments

Record:

- release version and Git commit;
- Python version and operating system;
- complete command line;
- all random seeds;
- task definitions and data splits;
- wall-clock time and hardware description;
- raw JSON output, not only summarized tables.

## Statistical interpretation

The current paired software experiment did not demonstrate superiority of hash
seeding over conventional pseudo-random seeding. This negative result is part of
the software's intended research value and must not be omitted from reports.

## Clean-environment verification

Create a fresh virtual environment, install from the tagged archive, run tests,
and compare output hashes.

## Archived release

VHSS/HashWave version 1.2.0 is permanently archived on Zenodo:

- version DOI: https://doi.org/10.5281/zenodo.21435911
- project concept DOI: https://doi.org/10.5281/zenodo.21435910
- GitHub release tag: `v1.2.0`

The tagged archive and version DOI are the reference artifacts for exact
reproduction. The `main` branch may contain documentation-only corrections
made after the tagged release. Such corrections must be recorded in the
changelog and release metadata and must not be described as changes to the
algorithm, datasets, tests, figures, package version, or reported numerical
results.
