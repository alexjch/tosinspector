.PHONY: lint typecheck test all clean

# Lint code with flake8
lint:
	flake8 --ignore E501 tosinspector/. tests/.

# Type check with mypy
typecheck:
	mypy tosinspector/ tests/

# Run tests with pytest
test:
	pytest

# Run all checks
all: lint typecheck test

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true