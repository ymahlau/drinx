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

The library lives in `src/drinx/` with four files:

- **`attribute.py`**: Defines `field`, `static_field`, `private_field`, `static_private_field`. All are thin wrappers around `dataclasses.field` that inject `jax_static=True/False` into the field's metadata dict. `private_*` variants set `init=False`. The unified `field()` function accepts a `static: bool` parameter directly.

- **`transform.py`**: Defines the `@dataclass` decorator and `_register_jax_tree`. The decorator wraps `dataclasses.dataclass` (always `frozen=True`) then registers the class as a JAX pytree. Flatten/unflatten split fields by `jax_static` metadata: static fields → `aux` (not traced), dynamic fields → `leaves` (traced). A `_jax_tree_registered` guard prevents double-registration.

- **`base.py`**: Defines `DataClass`, a base class alternative to the `@dataclass` decorator. Uses `@dataclass_transform` for type checker support and `__init_subclass__` to automatically apply the `dataclass` transform to any subclass. Also provides `aset(path, val)` for functional nested updates using path strings (e.g. `"a->b->[0]->['key']"`), `updated_copy(**kwargs)` as a wrapper around `dataclasses.replace`, and the `.at[key].set(val)` fluent API (via `_AtProxy`/`_AtIndexer`) that supports both path-based and mask-based updates.

- **`__init__.py`**: Re-exports `dataclass`, `field`, `static_field`, `private_field`, `static_private_field`, `DataClass`.

### Key design decisions

- All drinx dataclasses are **always frozen** (`frozen=True` is hardcoded). This is required for correctness as JAX pytree nodes must be immutable.
- The `jax_static` metadata key is the internal marker used to distinguish static vs. dynamic fields.
- Two usage patterns: decorator (`@drinx.dataclass`) or inheritance (`class Foo(DataClass)`). Both produce identically registered pytrees. `DataClass` methods (`aset`, `updated_copy`, `.at[].set()`) are only available with the inheritance pattern.
- The library has a single runtime dependency: `jax>=0.9.0`.
