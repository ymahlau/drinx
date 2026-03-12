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
    # Simple wrapper around dataclass.field which does exactly the same, except it has the additional static argument
    # to mark attributes
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
    # convenience wrapper: calls field above, with static argument set to True
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
    # convenience wrapper: field with init=False
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
    # convenience wrapper: field with init=False
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
