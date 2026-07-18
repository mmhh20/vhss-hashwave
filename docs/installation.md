# Installation

## Supported environments

Python 3.10 through 3.13 on Linux, macOS, and Windows are targeted by CI.
The runtime uses only the Python standard library.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Verify the installation:

```bash
vhss demo --generations 8 --beam 8 --per-parent 12 --seed install-check --json
python -m unittest discover -s tests -v
```

For an immutable research environment, record the Python interpreter version,
operating system, repository commit, tagged release, command line, and seeds.
