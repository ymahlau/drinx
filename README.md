![title image](https://github.com/ymahlau/drinx/blob/main/docs/source/_static/drinx.png?raw=true)

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://drinx.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/drinx)](https://pypi.org/project/drinx/)
[![codecov](https://codecov.io/gh/ymahlau/drinx/branch/main/graph/badge.svg)](https://codecov.io/gh/ymahlau/drinx)
[![Continuous integration](https://github.com/ymahlau/drinx/actions/workflows/cicd.yml/badge.svg?branch=main)](https://github.com/ymahlau/drinx/actions/workflows/cicd.yml/badge.svg?branch=main)

# Drinx: Dataclass Registry in JAX 🥂

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

## Quickstart

Below you can find some examples to get you quickly started with drinx.
But, beware, there are so much more features available, which are documented in detail in our [Documentation](https://drinx.readthedocs.io/en/latest/)

### Decorator style

Use `@drinx.dataclass` as a drop-in replacement for `@dataclasses.dataclass`.
The class is automatically frozen and registered as a JAX pytree:

```python
import jax
import jax.numpy as jnp
import drinx

@drinx.dataclass
class Params:
    weights: jax.Array
    bias: jax.Array

params = Params(weights=jnp.ones((3,)), bias=jnp.zeros((3,)))

# Works transparently with JAX transforms
doubled = jax.tree_util.tree_map(lambda x: x * 2, params)
```

### Static fields

Fields that should not be traced by JAX (e.g. shapes, dtypes, hyperparameters)
are marked with `static_field` or `field(static=True)`.  Changing a static
field triggers recompilation under `jit`:

```python
@drinx.dataclass
class Model:
    weights: jax.Array
    hidden_size: int = drinx.static_field(default=128)

@jax.jit
def forward(model, x):
    # hidden_size is a compile-time constant; weights are traced
    return model.weights[:model.hidden_size] @ x

model = Model(weights=jnp.ones((128, 32)))
```

### Inheritance style

Subclass `DataClass` instead of using the decorator.  The transform is applied
automatically — no `@dataclass` needed:

```python
class Model(drinx.DataClass):
    weights: jax.Array
    learning_rate: float = drinx.static_field(default=1e-3)

model = Model(weights=jnp.ones((10,)))
```

Dataclass options are forwarded via the class definition, or alternatively by using a combination of inheritance and decorator.

```python
class Config(drinx.DataClass, kw_only=True, order=True):
    hidden_size: int = drinx.static_field(default=128)
    num_layers: int = drinx.static_field(default=4)

# This is the recommended way: Typechecker will recognize the kw_only argument correctly
@drinx.dataclass(kw_only=True, order=True)
class Config(drinx.DataClass):
    hidden_size: int = drinx.static_field(default=128)
    num_layers: int = drinx.static_field(default=4)
```

### Functional updates with `aset`

Because drinx dataclasses are frozen, fields cannot be mutated in place.
`aset` performs a functional update and returns a new instance.  It supports
nested paths using `->` as a separator, integer indices `[n]`, and string
dictionary keys `['k']`.
Note that this function is only available when inheriting the `drinx.Dataclass`, but not from the decorator.

```python
class Inner(drinx.DataClass):
    w: jax.Array

class Outer(drinx.DataClass):
    inner: Inner
    bias: jax.Array

outer = Outer(inner=Inner(w=jnp.ones((3,))), bias=jnp.zeros((1,)))

# Update a top-level field
outer2 = outer.aset("bias", jnp.ones((1,)))

# Update a nested field
outer3 = outer.aset("inner->w", jnp.zeros((3,)))
```

### JAX transforms

Drinx dataclasses work with all JAX transforms out of the box:

```python
class State(drinx.DataClass):
    x: jax.Array
    step_size: float = drinx.static_field(default=0.1)

# jit
@jax.jit
def update(state):
    # updated_copy is convenience wrapper for altering top-level attributes
    return state.updated_copy(x=state.x - state.step_size)

def loss(state):
    return jnp.sum(state.x ** 2)

grads = jax.grad(loss)(State(x=jnp.array([1.0, 2.0, 3.0])))

@jax.vmap
def scale(state):
    return state.x * 2

batched = State(x=jnp.array([[1.0, 2.0], [3.0, 4.0]]))
result = scale(batched)  # shape (2, 2)
```

### Visualization

`tree_diagram` and `tree_summary` let you inspect any JAX pytree at a glance:

```python
class Encoder(drinx.DataClass):
    w: jax.Array
    b: jax.Array

class Model(drinx.DataClass):
    encoder: Encoder
    head: jax.Array

model = Model(encoder=Encoder(w=jnp.ones((16, 32)), b=jnp.zeros((16,))), head=jnp.ones((4, 16)))

print(drinx.tree_diagram(model))
# Model
# ├── .encoder:Encoder
# │   ├── .w=f32[16,32] ∈ [1.0, 1.0], μ=1.0, σ=0.0
# │   └── .b=f32[16] ∈ [0.0, 0.0], μ=0.0, σ=0.0
# └── .head=f32[4,16] ∈ [1.0, 1.0], μ=1.0, σ=0.0

print(drinx.tree_summary(model))
# ┌──────────────┬──────────┬───────┬────────┐
# │Name          │Type      │Count  │Size    │
# ├──────────────┼──────────┼───────┼────────┤
# │.encoder.w    │f32[16,32]│512    │2.00KB  │
# ├──────────────┼──────────┼───────┼────────┤
# │.encoder.b    │f32[16]   │16     │64.00B  │
# ├──────────────┼──────────┼───────┼────────┤
# │.head         │f32[4,16] │64     │256.00B │
# ├──────────────┼──────────┼───────┼────────┤
# │Σ             │Tree      │592    │2.31KB  │
# └──────────────┴──────────┴───────┴────────┘
```

## Documentation

For more examples and a detailed documentation, check out the API [here](https://drinx.readthedocs.io/en/latest/).


## Comparison to Alternatives

There exist a number of other libraries, which also integrate dataclass functionality with JAX.
The most similar libraries to drinx are [`jax_dataclass`](https://github.com/brentyi/jax_dataclasses) and [`pytreeclass`](https://github.com/ASEM000/pytreeclass), though `pytreeclass` is unfortunately no longer actively maintained.
Other libraries that are not specifically focusing on dataclasses, but have some functionality for them included, are the [`chex.dataclass`](https://github.com/google-deepmind/chex), [`flax.struct`](https://github.com/google/flax) or [`tjax.dataclass`](https://github.com/NeilGirdhar/tjax).

The main differences between the libraries are:
- **Static Fields:**: `drinx.field(static=True)` or `drinx.static_field()` can be used to mark dataclass attributes as static. The `pytreeclass` and `flax` library support a similar system. `jax_dataclass` supports static fields through marking an attribute as `Annotated[..., Static]`. `chex` and `tjax` do not support static attributes.
- **Attribute Updates**: `drinx` implements the `.at["attribute"].set()` syntax for functional updates of top-level attributes and `.aset()` for updates of nested structures. The `.at[].set()` syntax is also supported by `pytreeclass`, which heavily inspired our implementation. `tjax` implements a context manager which allows for updates of frozen classes, but this is non-functional and makes the usage in jit transforms difficult. `flax` implements a `.replace()` function for changing top-level attributes, but not nested updates.  `chex` only support updates through creating a new object.
- **Static Type Checking:** Both `drinx` and `jax_dataclass` are thin wrappers around the python dataclass and consequently have full type checking support. All other libraries have some limitations with regards for type checking, for example `kw_only` does not work in `tjax` or `flax`.
- **Visualization**: `drinx` implements some nice visualizations through `drinx.tree_diagram` and `drinx.tree_summary` which is heavily inspired by the `pytreeclass` library. Other libraries do not implement visualization tools out of the box.


## Citation

TODO: add citation once published

## Other links

Also check out my other repositories:
* [**💡 FDTDX**](https://github.com/ymahlau/fdtdx) - Electromagnetic FDTD Simulations in JAX. ![Stars](https://img.shields.io/github/stars/ymahlau/fdtdx?style=social)
* [**🔮 BONNI**](https://github.com/ymahlau/fdtdx) - Bayesian Optimization via Neural Network surrogates and Interior Point Optimization ![Stars](https://img.shields.io/github/stars/ymahlau/bonni?style=social)