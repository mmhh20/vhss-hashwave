PYTHON ?= python
.PHONY: test coverage demo benchmark verify

test:
	$(PYTHON) -m unittest discover -s tests -v

coverage:
	coverage erase
	coverage run --branch --source=hashwave -m unittest discover -s tests -v
	coverage report --fail-under=85

demo:
	$(PYTHON) -m hashwave demo --generations 20 --beam 16 --per-parent 32 --seed make --json

benchmark:
	$(PYTHON) -m hashwave benchmark --json

verify:
	$(PYTHON) scripts/verify_release.py
