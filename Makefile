.PHONY: setup test all placeholder help

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  setup       Install hcbm in editable mode"
	@echo "  test        Run acceptance tests with pytest"
	@echo "  all         Full pipeline with the real EEGPT checkpoint (scripts/run_all.sh)"
	@echo "  placeholder Full pipeline with a random-weight encoder (plumbing check)"

setup:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

all:
	scripts/run_all.sh

placeholder:
	BACKEND=template scripts/run_all.sh --random
