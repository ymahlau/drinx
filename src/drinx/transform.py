import jax
from dataclasses import dataclass as dataclass_orig
from dataclasses import fields, field as orig_field
from typing import TypeVar
from typing import dataclass_transform
from drinx.attribute import field, static_field, private_field, static_private_field
from typing import Callable, overload

T = TypeVar("T")


def _register_jax_tree(cls_: type[T]) -> type[T]:
    """Registers a class as a JAX Pytree, safely preventing double-registration."""
    # Guard: If already registered (e.g., by __init_subclass__), skip re-registering
    if getattr(cls_, "_jax_tree_registered", False):
        return cls_

    static_fields = [f.name for f in fields(cls_) if f.metadata.get("jax_static")]
    dynamic_fields = [f.name for f in fields(cls_) if not f.metadata.get("jax_static")]

    def flatten(obj):
        leaves = [getattr(obj, f) for f in dynamic_fields]
        aux = tuple(getattr(obj, f) for f in static_fields)
        return leaves, aux

    def unflatten(aux, leaves):
        kwargs = {**dict(zip(static_fields, aux)), **dict(zip(dynamic_fields, leaves))}
        return cls_(**kwargs)

    jax.tree_util.register_pytree_node(cls_, flatten, unflatten)
    cls_._jax_tree_registered = True  # ty:ignore[unresolved-attribute]
    return cls_


# Overload 1: For when the decorator is called WITHOUT arguments: @dataclass
@overload
def dataclass(
    cls: type[T],
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> type[T]: ...


# Overload 2: For when the decorator is called WITH arguments: @dataclass(kw_only=True)
@overload
def dataclass(
    cls: None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(
    field_specifiers=(
        orig_field,
        field,
        static_field,
        private_field,
        static_private_field,
    )
)
def dataclass(
    cls: type[T] | None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Decorator that converts a class into a frozen dataclass registered as a JAX pytree node.

    Wraps :func:`dataclasses.dataclass` with ``frozen=True`` and then calls
    :func:`_register_jax_tree` so the class can be used transparently with JAX
    transformations (``jit``, ``vmap``, ``grad``, etc.).

    Fields marked with :func:`static_field` (or ``field(static=True)``) are
    placed in the pytree auxiliary data and are excluded from JAX tracing.  All
    other fields become pytree leaves and are traced normally.

    Can be used with or without arguments::

        @drinx.dataclass
        class Params:
            weights: jax.Array
            lr: float = static_field(default=1e-3)

        @drinx.dataclass(kw_only=True)
        class Config:
            hidden_size: int = static_field(default=128)

    Args:
        cls: The class to decorate when used without arguments (``@dataclass``).
            ``None`` when called with arguments (``@dataclass(...)``).
        init: Generate ``__init__``.
        repr: Generate ``__repr__``.
        eq: Generate ``__eq__`` and ``__hash__``.
        order: Generate comparison methods (``<``, ``<=``, ``>``, ``>=``).
        unsafe_hash: Force generation of ``__hash__`` even when ``eq=True``.
        match_args: Set ``__match_args__`` for structural pattern matching.
        kw_only: Make all fields keyword-only in ``__init__``.
        slots: Generate ``__slots__``.
        weakref_slot: Add a ``__weakref__`` slot.

    Returns:
        The decorated class (when ``cls`` is provided), or a one-argument
        decorator (when called with keyword arguments only).

    Note:
        ``frozen=True`` is always enforced and cannot be overridden.  Mutability
        would break JAX's pytree contract.
    """

    # The wrapper handles the actual class modification
    def wrapper(cls_: type[T]) -> type[T]:
        # Detect if the class was already processed by __init_subclass__
        if getattr(cls_, "__dataclass_params__", None) is not None:
            # If so, it has generated __setattr__ and __delattr__ methods.
            # We must delete them from the class dictionary so the second
            # pass doesn't throw a "Cannot overwrite attribute" TypeError.
            if "__setattr__" in cls_.__dict__:
                delattr(cls_, "__setattr__")
            if "__delattr__" in cls_.__dict__:
                delattr(cls_, "__delattr__")

        decorator = dataclass_orig(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=True,
            match_args=match_args,
            kw_only=kw_only,
            slots=slots,
            weakref_slot=weakref_slot,
        )
        cls_ = decorator(cls_)
        return _register_jax_tree(cls_)

    if cls is None:
        return wrapper
    return wrapper(cls)
