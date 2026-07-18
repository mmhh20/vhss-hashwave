# Contributing

Contributions are welcome through public issues and pull requests after this
repository is published.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

## Contribution workflow

1. Open an issue describing the bug, research question, or proposed change.
2. Create a focused branch.
3. Add or update tests and documentation.
4. Run the full test and coverage commands.
5. Submit a pull request explaining design choices and reproducibility effects.

## Standards

- Preserve deterministic behavior for fixed seeds.
- Do not introduce unverified claims of quantum computation or hardware speedup.
- Separate software results from proposed physical-miner experiments.
- Add tests for cryptographic byte order, target comparisons, and serialization.
- Keep public APIs documented.

## Reporting research results

Commit raw machine-readable results together with the exact command, software
version, Python version, operating system, and random seeds. Do not overwrite
previously released results.
