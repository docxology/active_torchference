.PHONY: help install install-dev test test-verbose test-coverage examples clean lint format check-env

help:
	@echo "Active Torchference - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package in editable mode"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make check-env        Check Python environment and installation"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-verbose     Run tests with verbose output"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Examples:"
	@echo "  make examples         Run all examples with validation"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linting (flake8)"
	@echo "  make format           Format code (black)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove cache and build files"

install:
	@echo "Installing active_torchference..."
	pip install -e .
	@echo ""
	@make check-env

install-dev:
	@echo "Installing active_torchference with dev dependencies..."
	pip install -e ".[dev]"
	@echo ""
	@make check-env

check-env:
	@echo "Environment Check"
	@echo "================="
	@echo "Python version: $$(python --version)"
	@echo "Python path: $$(which python)"
	@echo "Python executable: $$(python -c 'import sys; print(sys.executable)')"
	@echo ""
	@echo "Checking package installation..."
	@python -c "import active_torchference; print('✓ Package installed successfully')" || (echo "✗ Package not installed - run 'make install'" && exit 1)
	@echo ""
	@echo "Package location:"
	@pip show active-torchference | grep Location || true
	@echo ""
	@echo "Ready to go! ✓"

test:
	@echo "Running test suite..."
	pytest

test-verbose:
	@echo "Running test suite (verbose)..."
	pytest -v

test-coverage:
	@echo "Running test suite with coverage..."
	pytest --cov=active_torchference --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

examples:
	@echo "Running all examples..."
	python run_all_examples.py

lint:
	@echo "Running linter..."
	flake8 active_torchference tests --max-line-length=100 --ignore=E203,W503

format:
	@echo "Formatting code..."
	black active_torchference tests examples --line-length=100

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build dist 2>/dev/null || true
	@echo "Cleanup complete!"

