# vLLM MUSA Platform Plugin - Makefile
# =====================================

.PHONY: help format lint test build clean publish install dev-install check all

# Default target
help:
	@echo "vLLM MUSA Platform Plugin - Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "    make dev-install  - Install package in development mode"
	@echo "    make install      - Install package"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make format       - Format code with isort and black"
	@echo "    make lint         - Run linters (ruff)"
	@echo "    make check        - Run format check without modifying files"
	@echo ""
	@echo "  Testing:"
	@echo "    make test         - Run all tests"
	@echo "    make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "  Build & Publish:"
	@echo "    make build        - Build wheel and sdist"
	@echo "    make publish      - Build and publish to PyPI"
	@echo "    make publish-test - Build and publish to TestPyPI"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean        - Remove build artifacts"
	@echo ""
	@echo "  Combined:"
	@echo "    make all          - format, lint, test, build"

# =============================================================================
# Development
# =============================================================================

dev-install:
	pip install -e ".[dev]"

install:
	pip install .

# =============================================================================
# Code Quality
# =============================================================================

# Format code with isort and black
# Preserves file ownership (useful when running in Docker as root)
format:
	@if [ "$$(id -u)" = "0" ]; then \
		isort vllm_musa_platform tests; \
		black vllm_musa_platform tests; \
		chown -R 1000:1000 vllm_musa_platform tests; \
	else \
		isort vllm_musa_platform tests; \
		black vllm_musa_platform tests; \
	fi

# Check formatting without modifying files
check:
	isort --check-only --diff vllm_musa_platform tests
	black --check --diff vllm_musa_platform tests

# Run linters
lint:
	ruff check vllm_musa_platform tests

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=vllm_musa_platform --cov-report=term-missing --cov-report=html

# =============================================================================
# Build & Publish
# =============================================================================

# Build wheel and source distribution
build: clean
	python -m build

# Publish to PyPI
publish: build
	python -m twine upload --repository pypi dist/*

# Publish to TestPyPI (for testing)
publish-test: build
	python -m twine upload --repository testpypi dist/*

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf vllm_musa_platform.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Combined Targets
# =============================================================================

all: format lint test build

