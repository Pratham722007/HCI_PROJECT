.PHONY: setup test preprocess embed train eval help

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  setup       Install hcbm in editable mode"
	@echo "  test        Run acceptance tests with pytest"
	@echo "  preprocess  Run EEG preprocessing and HDF5 generation"
	@echo "  embed       Extract and cache EEGPT embeddings"
	@echo "  train       Train H-CBM and baseline models on DENS"
	@echo "  eval        Run frozen evaluation on DEAP and DREAMER"

setup:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

preprocess:
	$(PYTHON) -m hcbm.preprocess

embed:
	$(PYTHON) -m hcbm.models.embed

train:
	$(PYTHON) -m hcbm.train

eval:
	$(PYTHON) -m hcbm.eval
