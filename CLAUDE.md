# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drinx is a small Python library ("Dataclass Registry in JAX") that wraps Python's standard `dataclasses` module to make dataclasses compatible with JAX transformations (e.g., `jit`, `vmap`, `grad`). It does this by registering each decorated class as a JAX pytree node, with support for marking fields as "static" (excluded from JAX tracing) or "dynamic" (included as leaves).

## Commands

This project uses `uv` for dependency management and virtual environments.

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_foo.py

# Run a single test
uv run pytest tests/test_foo.py::test_name

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uvx ty check --error-on-warning

# Build docs
uv run sphinx-build docs/source docs/build

# Live-reload docs
uv run sphinx-autobuild docs/source docs/build
```

Pre-commit hooks run ruff (lint + format), zizmor (GitHub Actions security), and `ty` (type checker) automatically on commit.

## Architecture

The entire library is three files in `src/drinx/`:

- **`field.py`**: Defines `field`, `static_field`, `private_field`, `static_private_field`. These are thin wrappers around `dataclasses.field` that inject `jax_static=True/False` into the field's metadata dict and optionally set `init=False` (for "private" variants).

- **`dataclass.py`**: Defines the `@dataclass` decorator. It wraps `dataclasses.dataclass` (always with `frozen=True`) and then calls `jax.tree_util.register_pytree_node` on the resulting class. The pytree flatten/unflatten functions split fields by `jax_static` metadata: static fields go into `aux` (auxiliary data, not traced), dynamic fields go into `leaves` (JAX arrays, traced).

- **`__init__.py`**: Re-exports `dataclass`, `field`, `static_field`, `private_field`, `static_private_field`.

### Key design decisions

- All drinx dataclasses are **always frozen** (`frozen=True` is hardcoded). This is required for correctness as JAX pytree nodes must be immutable.
- The `jax_static` metadata key is the internal marker used to distinguish static vs. dynamic fields.
- The library has a single runtime dependency: `jax>=0.9.0`.
