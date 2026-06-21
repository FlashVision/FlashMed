# Contributing to FlashMed

Thank you for your interest in contributing to FlashMed! This document provides guidelines for contributions.

## Development Setup

```bash
git clone https://github.com/FlashVision/FlashMed.git
cd FlashMed
pip install -e ".[dev,all]"
pre-commit install
```

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 120 characters
- Type hints are encouraged for public APIs
- Docstrings follow Google style

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass and linting is clean
4. Update documentation as needed
5. Submit a PR with a clear description

## Areas for Contribution

- New model architectures (e.g., Swin-UNet, TransUNet)
- Additional dataset loaders
- Privacy method improvements
- Performance optimizations
- Documentation improvements
- Bug fixes

## Medical AI Guidelines

- Never include real patient data in PRs
- Use synthetic/public datasets for examples
- Ensure privacy methods are properly tested
- Add appropriate disclaimers for clinical use

## Code of Conduct

Be respectful, constructive, and inclusive in all interactions.
