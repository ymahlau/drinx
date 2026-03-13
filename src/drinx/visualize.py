from __future__ import annotations
from drinx.jax_utils import is_traced

import numpy as np
import jax


def _fmt(x: float) -> str:
    """Format a float to 4 significant figures."""
    return f"{x:.2g}"


def visualize_leaf(val: int | float | complex | bool | np.ndarray | jax.Array) -> str:
    """Return a compact human-readable summary string for a JAX pytree leaf.

    Produces a type-annotated string representation with statistics appropriate
    for the value's kind:

    - **Python scalars** (``bool``, ``int``, ``float``, ``complex``): ``repr(val)``
    - **Tracers**: ``"<dtype>[<shape>] (Tracer)"``
    - **Scalar arrays** (0-d): ``"<dtype>[] <value>"``
    - **Empty arrays**: ``"<dtype>[<shape>] (empty)"``
    - **Boolean arrays**: ``"bool[<shape>] #T=<n_true>, #F=<n_false>"``
    - **Complex arrays**: ``"c<bits>[<shape>] |·| ∈ [min, max], μ=mean, σ=std"`` (stats on magnitude)
    - **Numeric arrays**: ``"<dtype>[<shape>] ∈ [min, max], μ=mean, σ=std"``

    The dtype string uses ``kind + bit-width`` notation (e.g. ``f32``, ``i64``, ``u8``, ``c128``),
    except booleans which are shown as ``bool``.

    Args:
        val: A pytree leaf — either a Python scalar or a NumPy/JAX array.

    Returns:
        A compact summary string describing the value's type, shape, and statistics.

    Raises:
        AssertionError: If ``val`` is not a supported type.
    """
    # 1. Handle Python built-in scalars
    if isinstance(val, (bool, int, float, complex)):
        return repr(val)

    assert isinstance(val, (np.ndarray, jax.Array)), f"Unsupported type: {type(val)}"

    dtype, shape = val.dtype, val.shape

    # 2. Build compact dtype string (NumPy's dtype.kind already returns 'f', 'i', 'u', 'c', 'b')
    dtype_str = "bool" if dtype.kind == "b" else f"{dtype.kind}{dtype.itemsize * 8}"
    prefix = (
        f"{dtype_str}[{','.join(map(str, shape))}]"  # ty:ignore[invalid-argument-type]
    )

    # 3. Handle Tracers
    if is_traced(val):
        return f"{prefix} (Tracer)"

    arr = np.asarray(val)

    # 4. Handle edge-case array shapes
    if arr.ndim == 0:
        return f"{prefix} {repr(arr.item())}"
    if arr.size == 0:
        return f"{prefix} (empty)"

    # 5. Handle Boolean arrays
    if dtype.kind == "b":
        n_true = int(arr.sum())
        return f"{prefix} #T={n_true}, #F={arr.size - n_true}"

    # 6. Consolidate Complex and Numeric arrays stats logic
    # Use magnitude for complex numbers, otherwise use the array as-is
    target = np.abs(arr) if dtype.kind == "c" else arr

    lo, hi = float(target.min()), float(target.max())
    # Calculate mean and std directly as floats to prevent overflow on smaller dtypes
    mu = float(target.mean(dtype=float))
    sigma = float(target.std(dtype=float))

    sym = "|·| " if dtype.kind == "c" else ""
    return f"{prefix} {sym}∈ [{_fmt(lo)}, {_fmt(hi)}], μ={_fmt(mu)}, σ={_fmt(sigma)}"
