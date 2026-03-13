from __future__ import annotations
from drinx.jax_utils import is_traced
from typing import Any

import dataclasses
import numpy as np
import jax
import jax.tree_util


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

    if not isinstance(val, (np.ndarray, jax.Array)):
        return repr(val)

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

    target = np.abs(arr) if dtype.kind == "c" else arr

    lo, hi = float(target.min()), float(target.max())
    # Calculate mean and std directly as floats to prevent overflow on smaller dtypes
    mu = float(target.mean(dtype=float))
    sigma = float(target.std(dtype=float))

    sym = "|·| " if dtype.kind == "c" else ""
    return f"{prefix} {sym}∈ [{_fmt(lo)}, {_fmt(hi)}], μ={_fmt(mu)}, σ={_fmt(sigma)}"


def _format_key(key: Any) -> str:
    """Convert a JAX path key to a display label."""
    if isinstance(key, jax.tree_util.GetAttrKey):
        return f".{key.name}"
    elif isinstance(key, jax.tree_util.SequenceKey):
        return f"[{key.idx}]"
    elif isinstance(key, jax.tree_util.DictKey):
        return f"['{key.key}']" if isinstance(key.key, str) else f"[{key.key}]"
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
        return f"[{key.key}]"
    else:
        return str(key)


def _get_one_level(
    node: Any, static_leaves: bool = False
) -> list[tuple[str, Any]] | None:
    """Get one level of children from a pytree node.

    Returns None if node is a leaf.
    """
    results, _ = jax.tree_util.tree_flatten_with_path(
        node, is_leaf=lambda x: x is not node
    )
    # A leaf has a single entry with an empty path
    if len(results) == 1 and len(results[0][0]) == 0:
        return None
    children = [(_format_key(path[0]), child) for path, child in results]
    if static_leaves and dataclasses.is_dataclass(node) and not isinstance(node, type):
        dynamic_dict = dict(children)
        ordered = []
        for f in dataclasses.fields(node):
            key = f".{f.name}"
            if f.metadata.get("jax_static"):
                ordered.append((key, getattr(node, f.name)))
            elif key in dynamic_dict:
                ordered.append((key, dynamic_dict[key]))
        children = ordered
    return children


def _build_lines(
    node: Any,
    depth: int,
    max_depth: int | None,
    prefix: str,
    lines: list[str],
    static_leaves: bool = False,
) -> None:
    children = _get_one_level(node, static_leaves)
    if children is None:
        return
    for i, (key_label, child) in enumerate(children):
        last = i == len(children) - 1
        connector = "└── " if last else "├── "
        child_children = _get_one_level(child, static_leaves)
        if child_children is None:
            lines.append(f"{prefix}{connector}{key_label}={visualize_leaf(child)}")
        elif max_depth is not None and depth + 1 >= max_depth:
            lines.append(f"{prefix}{connector}{key_label}:{type(child).__name__} ...")
        else:
            lines.append(f"{prefix}{connector}{key_label}:{type(child).__name__}")
            ext = "    " if last else "│   "
            _build_lines(
                child, depth + 1, max_depth, prefix + ext, lines, static_leaves
            )


def tree_diagram(
    tree: Any, max_depth: int | None = None, static_leaves: bool = False
) -> str:
    """Render a JAX pytree as an ASCII tree diagram.

    Args:
        tree: Any JAX pytree.
        max_depth: Maximum depth to expand. ``None`` means unlimited.
        static_leaves: If ``True``, show static fields of drinx dataclasses in
            declaration order, interleaved with dynamic fields.

    Returns:
        A multi-line string with the tree diagram.
    """
    lines = ["Tree"]
    _build_lines(tree, 0, max_depth, "", lines, static_leaves)
    return "\n".join(lines)
