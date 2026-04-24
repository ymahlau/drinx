"""JSON serialization and deserialization for drinx dataclasses."""

from __future__ import annotations

import dataclasses
import importlib
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _serialize(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-serializable structure.

    Supported types:
    - Python primitives: None, bool, int, float, str
    - list (serialized as JSON array)
    - tuple (tagged as ``{"__tuple__": [...]}`` to survive the JSON round-trip)
    - dict with str keys (tagged as ``{"__dict__": {...}}``)
    - jax.Array / np.ndarray
    - numpy/JAX dtype types (e.g. ``jnp.float32``, ``np.int32``)
    - dataclass instances (only ``init=True`` fields are included)

    Raises:
        TypeError: For non-string dict keys or unsupported value types.
        NotImplementedError: For nested / local dataclass classes whose
            ``__qualname__`` differs from ``__name__`` and therefore cannot be
            reconstructed via ``importlib``.
    """
    # None
    if obj is None:
        return None

    # bool must come before int (bool is a subclass of int)
    if isinstance(obj, bool):
        return obj

    # remaining primitives
    if isinstance(obj, (int, float, str)):
        return obj

    # list
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]

    # tuple
    if isinstance(obj, tuple):
        return {"__tuple__": [_serialize(v) for v in obj]}

    # dict (string keys only)
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError(
                "Only string-keyed dicts are supported for serialization; "
                f"got key types: {sorted({type(k).__name__ for k in obj if not isinstance(k, str)})}"
            )
        return {"__dict__": {k: _serialize(v) for k, v in obj.items()}}

    # JAX / numpy arrays
    if isinstance(obj, (jax.Array, np.ndarray)):
        arr = np.asarray(obj)
        if np.issubdtype(arr.dtype, np.complexfloating):
            data: Any = {"__real__": arr.real.tolist(), "__imag__": arr.imag.tolist()}
        else:
            data = arr.tolist()
        return {
            "__jax_array__": True,
            "__dtype__": arr.dtype.name,
            "__shape__": list(arr.shape),
            "__data__": data,
        }

    # JAX / numpy dtype TYPE objects (e.g. jnp.float32, np.int32)
    #    np.float32 etc. are subclasses of np.generic.
    #    jnp.float32 etc. in newer JAX use a custom _ScalarMeta metaclass and
    #    are NOT np.generic subclasses, so we check both.
    if (isinstance(obj, type) and issubclass(obj, np.generic)) or (
        type(obj).__name__ == "_ScalarMeta"
    ):
        return {"__jax_dtype__": obj.__name__}

    # dataclass instance
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        if cls.__qualname__ != cls.__name__:
            raise NotImplementedError(
                f"Cannot serialize nested or local class {cls.__qualname__!r}. "
                "Only module-level dataclasses are supported."
            )
        init_fields = {
            f.name: _serialize(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.init
        }
        return {
            "__module__": cls.__module__,
            "__name__": cls.__name__,
            "__value__": init_fields,
        }

    raise TypeError(
        f"Object of type {type(obj).__name__!r} is not JSON-serializable by drinx. "
        "For numpy scalar values, convert with float(x) or jnp.asarray(x) first."
    )


def _deserialize(data: Any) -> Any:
    """Recursively reconstruct a Python object from a JSON-decoded structure."""
    # Non-dict values: pass primitives through, recurse into lists
    if not isinstance(data, dict):
        if isinstance(data, list):
            return [_deserialize(v) for v in data]
        return data  # None, bool, int, float, str

    keys = set(data.keys())

    # tuple marker
    if keys == {"__tuple__"}:
        return tuple(_deserialize(v) for v in data["__tuple__"])

    # dict marker
    if keys == {"__dict__"}:
        return {k: _deserialize(v) for k, v in data["__dict__"].items()}

    # JAX dtype type
    if keys == {"__jax_dtype__"}:
        dtype_name = data["__jax_dtype__"]
        result = getattr(jnp, dtype_name, None)
        if result is None:
            raise ValueError(f"Unknown JAX dtype: {dtype_name!r}")
        return result

    # JAX / numpy array
    if (
        "__jax_array__" in data
        and "__dtype__" in data
        and "__shape__" in data
        and "__data__" in data
    ):
        dtype = np.dtype(data["__dtype__"])
        raw = data["__data__"]
        if isinstance(raw, dict) and "__real__" in raw and "__imag__" in raw:
            # complex array — reconstruct from split real/imag components
            arr = (np.array(raw["__real__"]) + 1j * np.array(raw["__imag__"])).astype(
                dtype
            )
        else:
            arr = np.array(raw, dtype=dtype)
        return jnp.asarray(arr)

    # dataclass
    if "__module__" in data and "__name__" in data and "__value__" in data:
        module = importlib.import_module(data["__module__"])
        cls = getattr(module, data["__name__"])
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{data['__name__']!r} is not a dataclass")
        kwargs = {k: _deserialize(v) for k, v in data["__value__"].items()}
        return cls(**kwargs)

    raise ValueError(
        f"Cannot deserialize dict with unrecognized keys: {keys}. "
        "The data may be malformed or from an incompatible version."
    )


def to_dict(obj: Any) -> dict:
    """Serialize a drinx dataclass instance to a JSON-serializable dict.

    Only ``init=True`` fields are included; private fields (``init=False``) are
    excluded and will be reconstructed from defaults or ``__post_init__`` when
    the object is later deserialized.

    Args:
        obj: A drinx (or standard) dataclass instance.

    Returns:
        A dict with ``"__module__"``, ``"__name__"``, and ``"__value__"`` keys.

    Raises:
        TypeError: If *obj* is not a dataclass instance, or if it contains
            field values of unsupported types.
        NotImplementedError: If *obj*'s class is a nested or local class.
    """
    result = _serialize(obj)
    if not isinstance(result, dict):
        raise TypeError(
            f"to_dict requires a dataclass instance, got {type(obj).__name__!r}"
        )
    return result


def from_dict(data: dict) -> Any:
    """Reconstruct a Python object from a dict produced by :func:`to_dict`.

    Args:
        data: A dict as returned by :func:`to_dict`.

    Returns:
        The reconstructed object.

    Raises:
        ModuleNotFoundError: If the stored module cannot be imported.
        AttributeError: If the stored class name does not exist in the module.
        TypeError: If the stored name refers to a non-dataclass.
        ValueError: If *data* has an unrecognized structure.
    """
    return _deserialize(data)


def to_json(obj: Any) -> str:
    """Serialize a drinx dataclass instance to a JSON string.

    Args:
        obj: A drinx (or standard) dataclass instance.

    Returns:
        A JSON string that can be passed to :func:`from_json`.
    """
    return json.dumps(to_dict(obj))


def from_json(json_str: str) -> Any:
    """Reconstruct a Python object from a JSON string produced by :func:`to_json`.

    Args:
        json_str: A JSON string as returned by :func:`to_json`.

    Returns:
        The reconstructed object.

    Raises:
        json.JSONDecodeError: If *json_str* is not valid JSON.
    """
    return from_dict(json.loads(json_str))
