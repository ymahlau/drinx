from typing import Any, Callable
from dataclasses import field as orig_field, MISSING


def field(
    *,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: Any = MISSING,
    static: bool = False,
) -> Any:
    """Define a dataclass field with optional JAX static marking.

    Thin wrapper around :func:`dataclasses.field` that injects the
    ``jax_static`` key into the field's metadata.  When ``static=True`` the
    field is excluded from JAX tracing and placed in the pytree auxiliary data;
    when ``static=False`` (the default) the field is a traced pytree leaf.

    Args:
        default: Default value for the field.
        default_factory: Zero-argument callable returning the default value.
            Mutually exclusive with *default*.
        init: Include the field in the generated ``__init__``.
        repr: Include the field in the generated ``__repr__``.
        hash: Include the field when computing ``__hash__``.  ``None`` defers
            to the value of *compare*.
        compare: Include the field in ``__eq__`` and ordering methods.
        metadata: Additional metadata dict merged with the ``jax_static`` entry.
        kw_only: Override the class-level ``kw_only`` setting for this field.
        static: When ``True``, mark the field as JAX-static (excluded from
            tracing).  Defaults to ``False``.

    Returns:
        A :class:`dataclasses.Field` descriptor (typed as ``Any`` for
        compatibility with type checkers).
    """
    metadata = metadata or {}
    metadata["jax_static"] = static

    return orig_field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def static_field(
    *,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: Any = MISSING,
) -> Any:
    """Define a JAX-static dataclass field.

    Convenience wrapper around :func:`field` with ``static=True`` pre-set.
    The field is excluded from JAX tracing and stored as pytree auxiliary data,
    meaning changes to it trigger recompilation under ``jit``.  Use this for
    configuration values, shapes, or other compile-time constants.

    Args:
        default: Default value for the field.
        default_factory: Zero-argument callable returning the default value.
        init: Include the field in the generated ``__init__``.
        repr: Include the field in the generated ``__repr__``.
        hash: Include the field in ``__hash__`` (``None`` defers to *compare*).
        compare: Include the field in ``__eq__`` and ordering methods.
        metadata: Additional metadata merged with the ``jax_static`` entry.
        kw_only: Override the class-level ``kw_only`` setting for this field.

    Returns:
        A :class:`dataclasses.Field` descriptor (typed as ``Any``).
    """
    return field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
        static=True,
    )


def private_field(
    *,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | Any = MISSING,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: Any = MISSING,
    static: bool = False,
) -> Any:
    """Define a private (non-init) dataclass field with optional JAX static marking.

    Convenience wrapper around :func:`field` with ``init=False`` pre-set.
    The field is excluded from ``__init__`` and must be assigned inside
    ``__post_init__`` or via a ``default``/``default_factory``.

    Args:
        default: Default value for the field.
        default_factory: Zero-argument callable returning the default value.
        repr: Include the field in the generated ``__repr__``.
        hash: Include the field in ``__hash__`` (``None`` defers to *compare*).
        compare: Include the field in ``__eq__`` and ordering methods.
        metadata: Additional metadata merged with the ``jax_static`` entry.
        kw_only: Override the class-level ``kw_only`` setting for this field.
        static: When ``True``, mark the field as JAX-static.  Defaults to
            ``False``.

    Returns:
        A :class:`dataclasses.Field` descriptor (typed as ``Any``).
    """
    return field(
        default=default,
        default_factory=default_factory,
        init=False,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
        static=static,
    )


def static_private_field(
    *,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | Any = MISSING,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: Any = MISSING,
) -> Any:
    """Define a private (non-init), JAX-static dataclass field.

    Convenience wrapper combining the behaviour of :func:`static_field` and
    :func:`private_field`: ``init=False`` and ``static=True`` are both
    pre-set.  The field is excluded from ``__init__`` and from JAX tracing,
    and must be assigned a value via ``default`` or ``default_factory``.

    Args:
        default: Default value for the field.
        default_factory: Zero-argument callable returning the default value.
        repr: Include the field in the generated ``__repr__``.
        hash: Include the field in ``__hash__`` (``None`` defers to *compare*).
        compare: Include the field in ``__eq__`` and ordering methods.
        metadata: Additional metadata merged with the ``jax_static`` entry.
        kw_only: Override the class-level ``kw_only`` setting for this field.

    Returns:
        A :class:`dataclasses.Field` descriptor (typed as ``Any``).
    """
    return field(
        default=default,
        default_factory=default_factory,
        init=False,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
        kw_only=kw_only,
        static=True,
    )
