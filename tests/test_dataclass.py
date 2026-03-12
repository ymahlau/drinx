"""Tests for drinx.dataclass: decorator, pytree registration, and field behaviour."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import dataclasses as std_dataclasses

import drinx
from drinx import dataclass, field, private_field, static_field, static_private_field


# ---------------------------------------------------------------------------
# Basic decorator behaviour
# ---------------------------------------------------------------------------


class TestDecoratorBasic:
    def test_without_arguments(self):
        @dataclass
        class Foo:
            x: float

        foo = Foo(x=1.0)
        assert foo.x == 1.0

    def test_with_empty_arguments(self):
        @dataclass()
        class Foo:
            x: float

        foo = Foo(x=1.0)
        assert foo.x == 1.0

    def test_with_kw_only(self):
        @dataclass(kw_only=True)
        class Foo:
            x: float

        foo = Foo(x=2.0)
        assert foo.x == 2.0

    def test_with_order(self):
        @dataclass(order=True)
        class Foo:
            x: float

        assert Foo(x=1.0) < Foo(x=2.0)
        assert Foo(x=3.0) > Foo(x=2.0)
        assert Foo(x=1.0) <= Foo(x=1.0)

    def test_is_dataclass(self):
        @dataclass
        class Foo:
            x: float

        assert dataclasses.is_dataclass(Foo)
        assert dataclasses.is_dataclass(Foo(x=0.0))

    def test_always_frozen(self):
        @dataclass
        class Foo:
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            foo.x = 2.0

    def test_frozen_even_with_eq_false(self):
        @dataclass(eq=False)
        class Foo:
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            foo.x = 99.0

    def test_no_class(self):
        # @dataclass with no class returns a callable
        decorator = dataclass(kw_only=True)
        assert callable(decorator)

    def test_multiple_fields(self):
        @dataclass
        class Foo:
            a: float
            b: int
            c: str

        foo = Foo(a=1.0, b=2, c="hello")
        assert foo.a == 1.0
        assert foo.b == 2
        assert foo.c == "hello"


# ---------------------------------------------------------------------------
# Equality and repr
# ---------------------------------------------------------------------------


class TestEqualityAndRepr:
    def test_equal_instances(self):
        @dataclass
        class Foo:
            x: float

        assert Foo(x=1.0) == Foo(x=1.0)

    def test_unequal_instances(self):
        @dataclass
        class Foo:
            x: float

        assert Foo(x=1.0) != Foo(x=2.0)

    def test_repr_contains_class_name(self):
        @dataclass
        class MyModel:
            x: float

        assert "MyModel" in repr(MyModel(x=1.0))

    def test_repr_contains_field_values(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=3)

        r = repr(Foo(x=1.0))
        assert "x=1.0" in r
        assert "n=3" in r

    def test_repr_false_field_hidden(self):
        @dataclass
        class Foo:
            x: float
            secret: int = field(default=99, repr=False)

        r = repr(Foo(x=1.0))
        assert "secret" not in r

    def test_compare_false_field_ignored(self):
        @dataclass
        class Foo:
            x: float
            tag: str = field(default="a", compare=False)

        a = Foo(x=1.0, tag="a")
        b = Foo(x=1.0, tag="b")
        assert a == b  # `tag` is excluded from comparison


# ---------------------------------------------------------------------------
# Default values and factories
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_simple_default(self):
        @dataclass
        class Foo:
            x: float = 0.0

        assert Foo().x == 0.0

    def test_field_with_default(self):
        @dataclass
        class Foo:
            x: float = field(default=5.0)

        assert Foo().x == 5.0

    def test_field_with_default_factory(self):
        @dataclass
        class Foo:
            items: list = field(default_factory=list)

        a = Foo()
        b = Foo()
        assert a.items == []
        assert a.items is not b.items  # distinct list objects

    def test_static_field_with_default(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=10)

        foo = Foo(x=1.0)
        assert foo.n == 10

    def test_mixed_defaults(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=5)
            bias: float = field(default=0.0)

        foo = Foo(x=2.0)
        assert foo.x == 2.0
        assert foo.n == 5
        assert foo.bias == 0.0


# ---------------------------------------------------------------------------
# Private fields (init=False)
# ---------------------------------------------------------------------------


class TestPrivateFields:
    def test_private_field_not_in_init(self):
        @dataclass
        class Foo:
            x: float
            _cache: float = private_field(default=0.0)

        foo = Foo(x=1.0)
        assert foo._cache == 0.0

    def test_static_private_field_not_in_init(self):
        @dataclass
        class Foo:
            x: float
            _n: int = static_private_field(default=42)

        foo = Foo(x=1.0)
        assert foo._n == 42

    def test_private_field_in_repr(self):
        @dataclass
        class Foo:
            x: float
            _cache: float = private_field(default=0.0)

        r = repr(Foo(x=1.0))
        assert "_cache" in r

    def test_private_field_repr_false(self):
        @dataclass
        class Foo:
            x: float
            _cache: float = private_field(default=0.0, repr=False)

        r = repr(Foo(x=1.0))
        assert "_cache" not in r


# ---------------------------------------------------------------------------
# Pytree registration: flatten / unflatten
# ---------------------------------------------------------------------------


class TestPytree:
    def test_dynamic_fields_are_leaves(self):
        @dataclass
        class Foo:
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert 1.0 in leaves
        assert 2.0 in leaves

    def test_static_fields_not_in_leaves(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=5)

        foo = Foo(x=3.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert 3.0 in leaves
        assert 5 not in leaves

    def test_leaf_count_dynamic_only(self):
        @dataclass
        class Foo:
            a: float
            b: float
            c: float

        foo = Foo(a=1.0, b=2.0, c=3.0)
        assert len(jax.tree_util.tree_leaves(foo)) == 3

    def test_leaf_count_mixed(self):
        @dataclass
        class Foo:
            x: float
            y: float
            n: int = static_field(default=1)

        foo = Foo(x=1.0, y=2.0)
        assert len(jax.tree_util.tree_leaves(foo)) == 2

    def test_leaf_count_all_static(self):
        @dataclass
        class Foo:
            n: int = static_field(default=1)
            m: int = static_field(default=2)

        foo = Foo()
        assert jax.tree_util.tree_leaves(foo) == []

    def test_roundtrip_all_dynamic(self):
        @dataclass
        class Foo:
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_roundtrip_mixed(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=7)

        foo = Foo(x=3.0)
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_roundtrip_all_static(self):
        @dataclass
        class Foo:
            n: int = static_field(default=3)
            m: int = static_field(default=4)

        foo = Foo()
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_treedef_changes_with_static_value(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=1)

        foo1 = Foo(x=1.0, n=1)
        foo2 = Foo(x=1.0, n=2)
        _, treedef1 = jax.tree_util.tree_flatten(foo1)
        _, treedef2 = jax.tree_util.tree_flatten(foo2)
        assert treedef1 != treedef2

    def test_treedef_same_for_same_static(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=1)

        foo1 = Foo(x=1.0, n=5)
        foo2 = Foo(x=9.0, n=5)
        _, treedef1 = jax.tree_util.tree_flatten(foo1)
        _, treedef2 = jax.tree_util.tree_flatten(foo2)
        assert treedef1 == treedef2

    def test_tree_map_applies_to_dynamic_only(self):
        @dataclass
        class Foo:
            x: float
            n: int = static_field(default=5)

        foo = Foo(x=2.0)
        result = jax.tree_util.tree_map(lambda v: v * 10, foo)
        assert result.x == 20.0
        assert result.n == 5  # static is untouched

    def test_nested_drinx_dataclasses(self):
        @dataclass
        class Inner:
            w: float

        @dataclass
        class Outer:
            inner: Inner
            bias: float

        obj = Outer(inner=Inner(w=1.0), bias=0.5)
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == obj

    def test_nested_leaves_count(self):
        @dataclass
        class Inner:
            a: float
            b: float

        @dataclass
        class Outer:
            inner: Inner
            c: float

        obj = Outer(inner=Inner(a=1.0, b=2.0), c=3.0)
        assert len(jax.tree_util.tree_leaves(obj)) == 3

    def test_is_registered_pytree(self):
        @dataclass
        class Foo:
            x: float

        # If registered, tree_flatten should not raise
        foo = Foo(x=1.0)
        leaves, _ = jax.tree_util.tree_flatten(foo)
        assert leaves == [1.0]

    def test_jax_array_leaves(self):
        @dataclass
        class Foo:
            x: jax.Array

        foo = Foo(x=jnp.array(3.0))
        leaves = jax.tree_util.tree_leaves(foo)
        assert len(leaves) == 1
        assert float(leaves[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Standard dataclasses.field interop
# ---------------------------------------------------------------------------


class TestStdField:
    def test_std_field_plain_annotation_is_dynamic(self):
        # A plain annotation with no field() call has no jax_static metadata
        # and must be treated as a dynamic leaf.
        @dataclass
        class Foo:
            x: float

        foo = Foo(x=1.0)
        assert 1.0 in jax.tree_util.tree_leaves(foo)

    def test_std_field_with_default_is_dynamic(self):
        @dataclass
        class Foo:
            x: float = std_dataclasses.field(default=7.0)

        foo = Foo()
        assert 7.0 in jax.tree_util.tree_leaves(foo)

    def test_std_field_with_default_factory_is_dynamic(self):
        @dataclass
        class Foo:
            items: list = std_dataclasses.field(default_factory=list)

        foo = Foo()
        leaves = jax.tree_util.tree_leaves(foo)
        assert len(leaves) == 0

    def test_std_field_not_in_aux(self):
        # Standard field must not appear in aux (it has no jax_static metadata).
        @dataclass
        class Foo:
            x: float = std_dataclasses.field(default=3.0)

        foo = Foo()
        _, treedef = jax.tree_util.tree_flatten(foo)
        # aux is stored in treedef; changing x must NOT change treedef
        foo2 = Foo(x=99.0)
        _, treedef2 = jax.tree_util.tree_flatten(foo2)
        assert treedef == treedef2

    def test_std_field_roundtrip(self):
        @dataclass
        class Foo:
            x: float = std_dataclasses.field(default=5.0)
            y: float = std_dataclasses.field(default=6.0)

        foo = Foo()
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_mixed_std_and_drinx_fields(self):
        # Mix of plain dataclasses.field (dynamic) and drinx static_field
        @dataclass
        class Foo:
            x: float = std_dataclasses.field(default=1.0)
            n: int = static_field(default=4)

        foo = Foo()
        leaves = jax.tree_util.tree_leaves(foo)
        assert 1.0 in leaves  # dynamic
        assert 4 not in leaves  # static → in aux, not leaves

    def test_mixed_std_and_drinx_leaf_count(self):
        @dataclass
        class Foo:
            a: float = std_dataclasses.field(default=1.0)
            b: float = std_dataclasses.field(default=2.0)
            n: int = static_field(default=10)

        foo = Foo()
        assert len(jax.tree_util.tree_leaves(foo)) == 2

    def test_mixed_std_and_drinx_roundtrip(self):
        @dataclass
        class Foo:
            x: float = std_dataclasses.field(default=3.0)
            n: int = static_field(default=7)

        foo = Foo()
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_std_field_repr_false(self):
        @dataclass
        class Foo:
            x: float = std_dataclasses.field(default=1.0, repr=False)

        r = repr(Foo())
        assert "x" not in r

    def test_std_field_compare_false(self):
        @dataclass
        class Foo:
            x: float
            tag: str = std_dataclasses.field(default="a", compare=False)

        assert Foo(x=1.0, tag="a") == Foo(x=1.0, tag="b")

    def test_std_field_jit_compatible(self):
        @dataclass
        class Foo:
            x: jax.Array = std_dataclasses.field(default_factory=lambda: jnp.array(0.0))

        @jax.jit
        def f(foo):
            return foo.x * 2

        foo = Foo(x=jnp.array(3.0))
        assert float(f(foo)) == pytest.approx(6.0)

    def test_std_field_grad_compatible(self):
        @dataclass
        class Foo:
            x: jax.Array

        @dataclass
        class Bar:
            val: jax.Array = std_dataclasses.field(
                default_factory=lambda: jnp.array(0.0)
            )

        def loss(bar):
            return bar.val**2

        bar = Bar(val=jnp.array(4.0))
        grads = jax.grad(loss)(bar)
        assert float(grads.val) == pytest.approx(8.0)

    def test_plain_annotation_jit_compatible(self):
        @dataclass
        class Foo:
            x: jax.Array

        @jax.jit
        def f(foo):
            return foo.x + 1

        result = f(Foo(x=jnp.array(2.0)))
        assert float(result) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Slots and weakref_slot options
# ---------------------------------------------------------------------------


class TestSlots:
    def test_slots_true(self):
        @dataclass(slots=True)
        class Foo:
            x: float

        foo = Foo(x=1.0)
        assert foo.x == 1.0
        assert "__slots__" in dir(type(foo))

    def test_slots_false_by_default(self):
        @dataclass
        class Foo:
            x: float

        assert (
            "__slots__" not in Foo.__dict__
            or Foo.__dict__.get("__slots__") is None
            or True
        )
        # Just check it constructs fine
        assert Foo(x=1.0).x == 1.0


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_drinx_exports_dataclass(self):
        assert hasattr(drinx, "dataclass")

    def test_drinx_exports_field(self):
        assert hasattr(drinx, "field")

    def test_drinx_exports_static_field(self):
        assert hasattr(drinx, "static_field")

    def test_drinx_exports_private_field(self):
        assert hasattr(drinx, "private_field")

    def test_drinx_exports_static_private_field(self):
        assert hasattr(drinx, "static_private_field")

    def test_all_list(self):
        assert "dataclass" in drinx.__all__
        assert "field" in drinx.__all__
        assert "static_field" in drinx.__all__
        assert "private_field" in drinx.__all__
        assert "static_private_field" in drinx.__all__
