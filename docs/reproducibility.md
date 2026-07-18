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
and compare output hashes. Release archives should be deposited in Zenodo or a
similar archival service after review changes are complete.
