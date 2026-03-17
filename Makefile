.PHONY: install, run, debug, clean, lint, lint-strict

PYTHON := $(shell command -v python3.11 2>/dev/null)
UV := $(shell command -v uv 2>/dev/null)
ARGS := $(wordlist 2, 999, $(MAKECMDGOALS))

install:
	@if [ -z "$(UV)" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH=$$PATH:$$HOME/.local/bin/uv; \
		UV=$$(command -v uv 2>/dev/null); \
	fi

	@if [ -z "$(PYTHON)" ]; then \
		uv python install 3.11; \
	fi

	@if [ ! -d "./.venv" ]; then \
		uv venv; \
	fi; \
	. ./.venv/bin/activate; \
	uv pip install mypy; \
	uv pip install flake8; \

run:
	@echo $(ARGS)
	@if [ ! -d .venv ]; then \
		echo "Run make install please"; \
		exit 1; \
	fi
	@sh -c '\
		. ./.venv/bin/activate; \
		uv run python -m src \
	'

debug:
	uv run python -m pdb main.py $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

lint:
	uv run flake8 ./src/*.py
	uv run mypy . --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

lint-strict: 
	uv run flake8 ./src/*.py
	uv run mypy . --strict