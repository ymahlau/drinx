import jax
from dataclasses import dataclass as dataclass_orig
from dataclasses import fields, field as orig_field
from typing import TypeVar
from typing import dataclass_transform
from drinx.field import field, static_field, private_field, static_private_field
from typing import Callable, overload

T = TypeVar("T")


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
    # The wrapper handles the actual class modification
    def wrapper(cls_: type[T]) -> type[T]:
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

        static_fields = [f.name for f in fields(cls_) if f.metadata.get("jax_static")]
        dynamic_fields = [
            f.name for f in fields(cls_) if not f.metadata.get("jax_static")
        ]

        def flatten(obj):
            leaves = [getattr(obj, f) for f in dynamic_fields]
            aux = tuple(getattr(obj, f) for f in static_fields)
            return leaves, aux

        def unflatten(aux, leaves):
            kwargs = {
                **dict(zip(static_fields, aux)),
                **dict(zip(dynamic_fields, leaves)),
            }
            return cls_(**kwargs)

        jax.tree_util.register_pytree_node(cls_, flatten, unflatten)
        return cls_

    # If called with args (e.g. @dataclass(kw_only=True)), return the wrapper
    if cls is None:
        return wrapper

    # If called as @dataclass, apply and return immediately
    return wrapper(cls)
