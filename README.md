![title image](https://github.com/ymahlau/drinx/blob/main/docs/source/_static/drinx.png?raw=true)

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://drinx.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/drinx)](https://pypi.org/project/drinx/)
[![codecov](https://codecov.io/gh/ymahlau/drinx/branch/main/graph/badge.svg)](https://codecov.io/gh/ymahlau/drinx)
[![Continuous integration](https://github.com/ymahlau/drinx/actions/workflows/cicd.yml/badge.svg?branch=main)](https://github.com/ymahlau/drinx/actions/workflows/cicd.yml/badge.svg?branch=main)

# Drinx: Dataclass Registry in JAX

Often it is useful to have structures in a program containing a mixture of JAX arrays and non-JAX types (e.g. strings, ...).
But, this makes it difficult to pass these objects through JAX transformations.
Drinx solves this by allowing dataclass fields to be declared as static.
Moreover, drinx introduces numerous quality-of-life features when working with dataclasses in JAX.

## Installation

You can install drinx simply via 

```bash
pip install drinx
```
If you want to use the GPU-acceleration from JAX, you can install afterwards:
```bash
pip install jax[cuda]
```

## Usage



