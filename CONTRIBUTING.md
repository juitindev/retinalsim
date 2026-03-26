# Contributing to RetinalSim

Thank you for your interest in contributing to RetinalSim! This document provides guidelines for contributing to this project.

## Ways to contribute

- **Bug reports** — open an issue describing the bug, expected behavior, and steps to reproduce
- **Feature requests** — open an issue describing the feature and its use case
- **Code contributions** — submit a pull request (see below)
- **Documentation** — improvements to README, docstrings, or examples
- **Validation** — comparing simulation output against published data or pulse2percept

## Development setup

```bash
git clone https://github.com/juit/retinalsim.git
cd retinalsim
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v --cov=retinalsim
```

All tests must pass before a PR will be merged.

## Pull request process

1. Fork the repository and create a feature branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Run the linter: `ruff check retinalsim/ tests/`
5. Update documentation if needed
6. Submit the pull request with a clear description of changes

## Code style

- Follow PEP 8, enforced by `ruff`
- Line length limit: 100 characters
- Use type hints for function signatures
- Include docstrings (NumPy style) for public functions
- Reference paper equations in comments (e.g., "Jansonius 2009 Eq.1")

## Model changes

Changes to the core model equations require:

1. A literature citation supporting the change
2. Updated comments referencing the equation source
3. Tests verifying the new behavior
4. If applicable, comparison against pulse2percept output

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## Questions?

Open an issue or reach out via the repository discussions tab.
