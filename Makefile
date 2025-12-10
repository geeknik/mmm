# Makefile for Melodic Metadata Massacrer

.PHONY: install dev-install test lint format clean run examples

# Default target
all: install

# Install package and dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Install in development mode with test dependencies
dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v --cov=mmm --cov-report=html --cov-report=term

# Run tests with coverage
test-cov:
	python -m pytest tests/ -v --cov=mmm --cov-report=html

# Lint code
lint:
	flake8 mmm/ tests/
	mypy mmm/

# Format code
format:
	black mmm/ tests/
	isort mmm/ tests/

# Check formatting
check-format:
	black --check mmm/ tests/
	isort --check-only mmm/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python setup.py sdist bdist_wheel

# Run the application
run:
	python -m mmm.cli --help

# Example usage
examples:
	python -m mmm.cli --help
	@echo "Example: mmm obliterate input.mp3 --paranoid --verify"

# Create test audio files for testing
test-audio:
	python -c "
import numpy as np
import soundfile as sf
from pathlib import Path

# Create test directory
Path('test_audio').mkdir(exist_ok=True)

# Generate test audio
sample_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Create test files
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine
sf.write('test_audio/sine_440hz.wav', audio, sample_rate)

# Create stereo version
stereo = np.column_stack([audio, audio * 0.8])
sf.write('test_audio/stereo_test.wav', stereo, sample_rate)

# Create more complex signal
complex_audio = (0.3 * np.sin(2 * np.pi * 220 * t) +
                 0.2 * np.sin(2 * np.pi * 440 * t) +
                 0.1 * np.sin(2 * np.pi * 880 * t))
sf.write('test_audio/complex_signal.wav', complex_audio, sample_rate)

print('Test audio files created in test_audio/')
"

# Demo with test files
demo: test-audio
	python -m mmm.cli analyze test_audio/sine_440hz.wav
	python -m mmm.cli obliterate test_audio/sine_440hz.wav --verify
	python -m mmm.cli massacre test_audio/ --output-dir test_audio/cleaned/

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Run full CI checks
ci: check-format lint test

# Docker build (if using Docker)
docker-build:
	docker build -t mmm:latest .

# Docker run
docker-run:
	docker run -v $(PWD)/test_audio:/data mmm:latest obliterate /data/sine_440hz.wav

# Release preparation
release-check: clean format lint test
	python -m build

# Install git hooks (optional)
git-hooks:
	cp scripts/git-hooks/* .git/hooks/
	chmod +x .git/hooks/*

# Documentation
docs:
	python -c "
import mmm
print('MMM Documentation')
print('==================')
print('Version:', mmm.__version__)
print('Description:', mmm.__description__)
print()
print('Commands:')
print('  mmm obliterate INPUT_FILE     - Remove all metadata and watermarks')
print('  mmm massacre DIRECTORY       - Batch process directory')
print('  mmm analyze INPUT_FILE        - Analyze without modifying')
print('  mmm config                   - Show configuration')
print()
print('Examples:')
print('  mmm obliterate music.mp3 --paranoid --verify')
print('  mmm massacre /music/dir --workers 8')
print()
"

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install package and dependencies"
	@echo "  dev-install   - Install in development mode"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  lint          - Lint code with flake8 and mypy"
	@echo "  format        - Format code with black and isort"
	@echo "  check-format  - Check code formatting"
	@echo "  clean         - Remove build artifacts"
	@echo "  build         - Build package"
	@echo "  run           - Run the application"
	@echo "  examples      - Show example commands"
	@echo "  test-audio    - Create test audio files"
	@echo "  demo          - Run demo with test files"
	@echo "  ci            - Run full CI checks"
	@echo "  docs          - Show documentation"
	@echo "  help          - Show this help message"