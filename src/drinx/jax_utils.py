from typing import Any
from jax import core


def is_traced(x: Any) -> bool:
    """
    Checks if an object is a JAX Tracer.

    In JAX, tracers are used during transformations (like `jit`, `grad`, or `vmap`)
    to represent abstract values rather than concrete arrays. This function
    identifies if the input is currently being tracked by the JAX dispatcher.

    Args:
        x: The object to check.

    Returns:
        True if `x` is a `jax.core.Tracer`, False otherwise.
    """
    return isinstance(x, core.Tracer)
