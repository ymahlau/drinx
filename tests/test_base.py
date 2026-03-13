"""Exhaustive tests for drinx.DataClass base class."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import dataclasses as std_dataclasses

import drinx
from drinx import DataClass, field, private_field, static_field, static_private_field


# ---------------------------------------------------------------------------
# Basic subclassing
# ---------------------------------------------------------------------------


class TestBasicSubclassing:
    def test_simple_subclass(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        assert foo.x == 1.0

    def test_multiple_fields(self):
        class Foo(DataClass):
            a: float
            b: int
            c: str

        foo = Foo(a=1.0, b=2, c="hello")
        assert foo.a == 1.0
        assert foo.b == 2
        assert foo.c == "hello"

    def test_is_dataclass_class(self):
        class Foo(DataClass):
            x: float

        assert dataclasses.is_dataclass(Foo)

    def test_is_dataclass_instance(self):
        class Foo(DataClass):
            x: float

        assert dataclasses.is_dataclass(Foo(x=1.0))

    def test_fields_accessible(self):
        class Foo(DataClass):
            x: float
            y: int

        fields = dataclasses.fields(Foo)
        field_names = [f.name for f in fields]
        assert "x" in field_names
        assert "y" in field_names

    def test_no_decorator_needed(self):
        # Inheritance alone is sufficient — no @dataclass needed
        class Model(DataClass):
            w: float
            b: float

        m = Model(w=0.5, b=0.1)
        assert m.w == 0.5
        assert m.b == 0.1


# ---------------------------------------------------------------------------
# Always frozen
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_cannot_set_attribute(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            foo.x = 2.0

    def test_cannot_set_new_attribute(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            foo.new_attr = 99  # type: ignore[attr-defined]

    def test_cannot_delete_attribute(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            del foo.x

    def test_frozen_with_kw_only(self):
        class Foo(DataClass, kw_only=True):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            foo.x = 2.0


# ---------------------------------------------------------------------------
# Class keyword arguments (passed via __init_subclass__)
# ---------------------------------------------------------------------------


class TestClassKeywordArgs:
    def test_kw_only_true(self):
        class Foo(DataClass, kw_only=True):
            x: float

        # kw_only means positional args are not allowed
        with pytest.raises(TypeError):
            Foo(1.0)
        foo = Foo(x=1.0)
        assert foo.x == 1.0

    def test_kw_only_false_default(self):
        class Foo(DataClass):
            x: float

        # Default allows positional args
        foo = Foo(1.0)
        assert foo.x == 1.0

    def test_order_true(self):
        class Foo(DataClass, order=True):
            x: float

        assert Foo(x=1.0) < Foo(x=2.0)
        assert Foo(x=3.0) > Foo(x=2.0)
        assert Foo(x=1.0) <= Foo(x=1.0)
        assert Foo(x=1.0) >= Foo(x=1.0)

    def test_order_false_default(self):
        class Foo(DataClass):
            x: float

        with pytest.raises(TypeError):
            Foo(x=1.0) < Foo(x=2.0)  # type: ignore[operator]

    def test_eq_true_default(self):
        class Foo(DataClass):
            x: float

        assert Foo(x=1.0) == Foo(x=1.0)
        assert Foo(x=1.0) != Foo(x=2.0)

    def test_eq_false(self):
        class Foo(DataClass, eq=False):
            x: float

        # Without eq, instances compare by identity
        foo = Foo(x=1.0)
        assert foo != Foo(x=1.0)  # different objects
        assert foo == foo  # same object

    def test_repr_true_default(self):
        class Foo(DataClass):
            x: float

        r = repr(Foo(x=1.0))
        assert "Foo" in r
        assert "x=1.0" in r

    def test_repr_false(self):
        class Foo(DataClass, repr=False):
            x: float

        r = repr(Foo(x=1.0))
        # Default object repr does not contain field values
        assert "x=1.0" not in r

    def test_init_false(self):
        # With init=False, the generated __init__ is suppressed
        # The class must define its own __init__
        class Foo(DataClass, init=False):
            x: float

            def __init__(self, x: float):
                object.__setattr__(self, "x", x)

        foo = Foo(1.0)
        assert foo.x == 1.0

    def test_match_args_false(self):
        class Foo(DataClass, match_args=False):
            x: float

        # match_args=False means no __match_args__ generated
        assert not hasattr(Foo, "__match_args__") or Foo.__match_args__ == ()


# ---------------------------------------------------------------------------
# Default values and factories
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_simple_default(self):
        class Foo(DataClass):
            x: float = 1.0

        assert Foo().x == 1.0

    def test_field_with_default(self):
        class Foo(DataClass):
            x: float = field(default=5.0)

        assert Foo().x == 5.0

    def test_field_with_default_factory(self):
        class Foo(DataClass):
            items: tuple = field(default_factory=tuple)

        a = Foo()
        b = Foo()
        assert a.items == ()
        assert a.items is not b.items or a.items == b.items  # both valid for tuples

    def test_static_field_with_default(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=10)

        foo = Foo(x=1.0)
        assert foo.n == 10

    def test_mixed_defaults(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=5)
            bias: float = field(default=0.0)

        foo = Foo(x=2.0)
        assert foo.x == 2.0
        assert foo.n == 5
        assert foo.bias == 0.0

    def test_static_field_overridable(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=3)

        foo = Foo(x=1.0, n=7)
        assert foo.n == 7


# ---------------------------------------------------------------------------
# Private fields
# ---------------------------------------------------------------------------


class TestPrivateFields:
    def test_private_field_not_in_init(self):
        class Foo(DataClass):
            x: float
            _cache: float = private_field(default=0.0)

        foo = Foo(x=1.0)
        assert foo._cache == 0.0

    def test_static_private_field_not_in_init(self):
        class Foo(DataClass):
            x: float
            _n: int = static_private_field(default=42)

        foo = Foo(x=1.0)
        assert foo._n == 42

    def test_private_field_in_repr_by_default(self):
        class Foo(DataClass):
            x: float
            _cache: float = private_field(default=0.0)

        r = repr(Foo(x=1.0))
        assert "_cache" in r

    def test_private_field_repr_false(self):
        class Foo(DataClass):
            x: float
            _cache: float = private_field(default=0.0, repr=False)

        r = repr(Foo(x=1.0))
        assert "_cache" not in r

    def test_private_field_is_dynamic(self):
        class Foo(DataClass):
            x: float
            _y: float = private_field(default=2.0)

        foo = Foo(x=1.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert 1.0 in leaves
        assert 2.0 in leaves

    def test_static_private_field_not_in_leaves(self):
        class Foo(DataClass):
            x: float
            _n: int = static_private_field(default=99)

        foo = Foo(x=1.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert 99 not in leaves


# ---------------------------------------------------------------------------
# Pytree registration: flatten / unflatten
# ---------------------------------------------------------------------------


class TestPytree:
    def test_dynamic_fields_are_leaves(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert 1.0 in leaves
        assert 2.0 in leaves

    def test_static_fields_not_in_leaves(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=5)

        foo = Foo(x=3.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert 3.0 in leaves
        assert 5 not in leaves

    def test_leaf_count_dynamic_only(self):
        class Foo(DataClass):
            a: float
            b: float
            c: float

        foo = Foo(a=1.0, b=2.0, c=3.0)
        assert len(jax.tree_util.tree_leaves(foo)) == 3

    def test_leaf_count_mixed(self):
        class Foo(DataClass):
            x: float
            y: float
            n: int = static_field(default=1)

        foo = Foo(x=1.0, y=2.0)
        assert len(jax.tree_util.tree_leaves(foo)) == 2

    def test_leaf_count_all_static(self):
        class Foo(DataClass):
            n: int = static_field(default=1)
            m: int = static_field(default=2)

        foo = Foo()
        assert jax.tree_util.tree_leaves(foo) == []

    def test_roundtrip_all_dynamic(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_roundtrip_mixed(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=7)

        foo = Foo(x=3.0)
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_roundtrip_all_static(self):
        class Foo(DataClass):
            n: int = static_field(default=3)
            m: int = static_field(default=4)

        foo = Foo()
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_treedef_changes_with_static_value(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=1)

        foo1 = Foo(x=1.0, n=1)
        foo2 = Foo(x=1.0, n=2)
        _, treedef1 = jax.tree_util.tree_flatten(foo1)
        _, treedef2 = jax.tree_util.tree_flatten(foo2)
        assert treedef1 != treedef2

    def test_treedef_same_for_same_static(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=1)

        foo1 = Foo(x=1.0, n=5)
        foo2 = Foo(x=9.0, n=5)
        _, treedef1 = jax.tree_util.tree_flatten(foo1)
        _, treedef2 = jax.tree_util.tree_flatten(foo2)
        assert treedef1 == treedef2

    def test_tree_map_applies_to_dynamic_only(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=5)

        foo = Foo(x=2.0)
        result = jax.tree_util.tree_map(lambda v: v * 10, foo)
        assert result.x == 20.0
        assert result.n == 5  # static is untouched

    def test_is_registered_pytree(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        leaves, _ = jax.tree_util.tree_flatten(foo)
        assert leaves == [1.0]

    def test_jax_array_leaves(self):
        class Foo(DataClass):
            x: jax.Array

        foo = Foo(x=jnp.array(3.0))
        leaves = jax.tree_util.tree_leaves(foo)
        assert len(leaves) == 1
        assert float(leaves[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# JAX transformations
# ---------------------------------------------------------------------------


class TestJaxTransformations:
    def test_jit(self):
        class Foo(DataClass):
            x: jax.Array

        @jax.jit
        def f(foo):
            return foo.x * 2

        foo = Foo(x=jnp.array(3.0))
        assert float(f(foo)) == pytest.approx(6.0)

    def test_jit_with_static_field(self):
        class Foo(DataClass):
            x: jax.Array
            scale: int = static_field(default=2)

        @jax.jit
        def f(foo):
            return foo.x * foo.scale

        foo = Foo(x=jnp.array(3.0))
        assert float(f(foo)) == pytest.approx(6.0)

    def test_grad(self):
        class Foo(DataClass):
            x: jax.Array

        def loss(foo):
            return foo.x**2

        foo = Foo(x=jnp.array(4.0))
        grads = jax.grad(loss)(foo)
        assert float(grads.x) == pytest.approx(8.0)

    def test_grad_with_static(self):
        class Foo(DataClass):
            x: jax.Array
            n: int = static_field(default=2)

        def loss(foo):
            return foo.x**foo.n

        foo = Foo(x=jnp.array(3.0))
        grads = jax.grad(loss)(foo)
        assert float(grads.x) == pytest.approx(6.0)

    def test_vmap(self):
        class Foo(DataClass):
            x: jax.Array

        @jax.vmap
        def f(foo):
            return foo.x * 2

        batch = Foo(x=jnp.array([1.0, 2.0, 3.0]))
        result = f(batch)
        assert result.shape == (3,)
        assert float(result[0]) == pytest.approx(2.0)
        assert float(result[1]) == pytest.approx(4.0)
        assert float(result[2]) == pytest.approx(6.0)

    def test_jit_recompiles_on_static_change(self):
        call_count = 0

        class Foo(DataClass):
            x: jax.Array
            mode: str = static_field(default="add")

        @jax.jit
        def f(foo):
            nonlocal call_count
            call_count += 1
            return foo.x

        f(Foo(x=jnp.array(1.0), mode="add"))
        f(Foo(x=jnp.array(2.0), mode="add"))  # same static → no recompile
        count_after_two = call_count
        f(Foo(x=jnp.array(3.0), mode="mul"))  # different static → recompile
        assert call_count > count_after_two


# ---------------------------------------------------------------------------
# Inheritance: DataClass subclassing a DataClass subclass
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_subclass_of_dataclass_subclass(self):
        class Base(DataClass):
            x: float

        class Child(Base):
            y: float

        child = Child(x=1.0, y=2.0)
        assert child.x == 1.0
        assert child.y == 2.0

    def test_child_is_dataclass(self):
        class Base(DataClass):
            x: float

        class Child(Base):
            y: float

        assert dataclasses.is_dataclass(Child)

    def test_child_is_not_independently_registered_pytree(self):
        # Known limitation: Child inherits `_jax_tree_registered = True` from
        # Base, so _register_jax_tree skips registration for Child. JAX
        # therefore treats Child instances as opaque leaves rather than
        # flattening them through their own fields.
        class Base(DataClass):
            x: float

        class Child(Base):
            y: float

        child = Child(x=1.0, y=2.0)
        leaves = jax.tree_util.tree_leaves(child)
        # Child is unregistered → treated as a single leaf
        assert leaves == [child]

    def test_child_frozen(self):
        class Base(DataClass):
            x: float

        class Child(Base):
            y: float

        child = Child(x=1.0, y=2.0)
        with pytest.raises(
            (dataclasses.FrozenInstanceError, TypeError, AttributeError)
        ):
            child.x = 99.0

    def test_child_inherits_methods(self):
        class Base(DataClass):
            x: float

            def double(self):
                return self.x * 2

        class Child(Base):
            y: float

        child = Child(x=3.0, y=0.0)
        assert child.double() == 6.0

    def test_three_level_inheritance(self):
        class A(DataClass):
            a: float

        class B(A):
            b: float

        class C(B):
            c: float

        obj = C(a=1.0, b=2.0, c=3.0)
        assert obj.a == 1.0
        assert obj.b == 2.0
        assert obj.c == 3.0

    def test_child_with_static_fields_is_opaque_leaf(self):
        # Same limitation as test_child_is_not_independently_registered_pytree:
        # Child is not re-registered, so static_field on Child has no pytree
        # effect — the whole Child instance is a single leaf.
        class Base(DataClass):
            x: float

        class Child(Base):
            y: float
            n: int = static_field(default=3)

        child = Child(x=1.0, y=2.0)
        leaves = jax.tree_util.tree_leaves(child)
        assert leaves == [child]

    def test_child_roundtrip(self):
        class Base(DataClass):
            x: float

        class Child(Base):
            y: float

        child = Child(x=1.0, y=2.0)
        leaves, treedef = jax.tree_util.tree_flatten(child)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == child


# ---------------------------------------------------------------------------
# Nested DataClass instances
# ---------------------------------------------------------------------------


class TestNested:
    def test_nested_as_field(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner
            bias: float

        obj = Outer(inner=Inner(w=1.0), bias=0.5)
        leaves = jax.tree_util.tree_leaves(obj)
        assert len(leaves) == 2

    def test_nested_roundtrip(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner
            bias: float

        obj = Outer(inner=Inner(w=1.0), bias=0.5)
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == obj

    def test_nested_tree_map(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner
            bias: float

        obj = Outer(inner=Inner(w=2.0), bias=3.0)
        result = jax.tree_util.tree_map(lambda v: v * 10, obj)
        assert result.inner.w == 20.0
        assert result.bias == 30.0

    def test_deeply_nested(self):
        class A(DataClass):
            val: float

        class B(DataClass):
            a: A
            x: float

        class C(DataClass):
            b: B
            y: float

        obj = C(b=B(a=A(val=1.0), x=2.0), y=3.0)
        leaves = jax.tree_util.tree_leaves(obj)
        assert len(leaves) == 3


# ---------------------------------------------------------------------------
# Equality and repr
# ---------------------------------------------------------------------------


class TestEqualityAndRepr:
    def test_equal_instances(self):
        class Foo(DataClass):
            x: float

        assert Foo(x=1.0) == Foo(x=1.0)

    def test_unequal_instances(self):
        class Foo(DataClass):
            x: float

        assert Foo(x=1.0) != Foo(x=2.0)

    def test_repr_contains_class_name(self):
        class MyModel(DataClass):
            x: float

        assert "MyModel" in repr(MyModel(x=1.0))

    def test_repr_contains_field_values(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=3)

        r = repr(Foo(x=1.0))
        assert "x=1.0" in r
        assert "n=3" in r

    def test_repr_false_field_hidden(self):
        class Foo(DataClass):
            x: float
            secret: int = field(default=99, repr=False)

        r = repr(Foo(x=1.0))
        assert "secret" not in r

    def test_compare_false_field_ignored(self):
        class Foo(DataClass):
            x: float
            tag: str = field(default="a", compare=False)

        a = Foo(x=1.0, tag="a")
        b = Foo(x=1.0, tag="b")
        assert a == b

    def test_hash_consistent(self):
        class Foo(DataClass):
            x: float

        foo1 = Foo(x=1.0)
        foo2 = Foo(x=1.0)
        # frozen dataclasses are hashable
        assert hash(foo1) == hash(foo2)

    def test_usable_as_dict_key(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        d = {foo: "value"}
        assert d[Foo(x=1.0)] == "value"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_dataclass_exported(self):
        assert hasattr(drinx, "DataClass")

    def test_dataclass_in_all(self):
        assert "DataClass" in drinx.__all__

    def test_dataclass_is_class(self):
        assert isinstance(DataClass, type)

    def test_import_directly(self):
        from drinx import DataClass as DC

        assert DC is DataClass


# ---------------------------------------------------------------------------
# Standard dataclasses.field interop
# ---------------------------------------------------------------------------


class TestStdFieldInterop:
    def test_std_field_is_dynamic(self):
        class Foo(DataClass):
            x: float = std_dataclasses.field(default=7.0)

        foo = Foo()
        assert 7.0 in jax.tree_util.tree_leaves(foo)

    def test_std_field_not_in_aux(self):
        class Foo(DataClass):
            x: float = std_dataclasses.field(default=3.0)

        foo = Foo()
        foo2 = Foo(x=99.0)
        _, treedef1 = jax.tree_util.tree_flatten(foo)
        _, treedef2 = jax.tree_util.tree_flatten(foo2)
        assert treedef1 == treedef2

    def test_mixed_std_and_drinx_fields(self):
        class Foo(DataClass):
            x: float = std_dataclasses.field(default=1.0)
            n: int = static_field(default=4)

        foo = Foo()
        leaves = jax.tree_util.tree_leaves(foo)
        assert 1.0 in leaves
        assert 4 not in leaves

    def test_std_field_compare_false(self):
        class Foo(DataClass):
            x: float
            tag: str = std_dataclasses.field(default="a", compare=False)

        assert Foo(x=1.0, tag="a") == Foo(x=1.0, tag="b")

    def test_std_field_repr_false(self):
        class Foo(DataClass):
            x: float = std_dataclasses.field(default=1.0, repr=False)

        r = repr(Foo())
        assert "x" not in r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_class(self):
        class Foo(DataClass):
            pass

        foo = Foo()
        assert jax.tree_util.tree_leaves(foo) == []

    def test_many_fields(self):
        class Foo(DataClass):
            a: float
            b: float
            c: float
            d: float
            e: float

        foo = Foo(a=1.0, b=2.0, c=3.0, d=4.0, e=5.0)
        leaves = jax.tree_util.tree_leaves(foo)
        assert len(leaves) == 5

    def test_boolean_field(self):
        class Foo(DataClass):
            flag: bool = static_field(default=True)
            x: float = 0.0

        foo = Foo()
        assert foo.flag is True
        assert True not in jax.tree_util.tree_leaves(foo)

    def test_string_static_field(self):
        class Foo(DataClass):
            x: float
            name: str = static_field(default="default")

        foo = Foo(x=1.0, name="test")
        leaves = jax.tree_util.tree_leaves(foo)
        assert "test" not in leaves
        _, treedef1 = jax.tree_util.tree_flatten(foo)
        _, treedef2 = jax.tree_util.tree_flatten(Foo(x=2.0, name="other"))
        assert treedef1 != treedef2

    def test_tuple_dynamic_field(self):
        class Foo(DataClass):
            x: float
            coords: tuple = field(default_factory=lambda: (0.0, 0.0))

        foo = Foo(x=1.0)
        leaves = jax.tree_util.tree_leaves(foo)
        # Tuples are themselves pytree nodes, so their elements become leaves
        assert 1.0 in leaves

    def test_none_default(self):
        class Foo(DataClass):
            x: float
            extra: object = field(default=None)

        foo = Foo(x=1.0)
        assert foo.extra is None

    def test_class_with_only_static_fields_is_pytree(self):
        class Foo(DataClass):
            n: int = static_field(default=1)

        foo = Foo()
        leaves, treedef = jax.tree_util.tree_flatten(foo)
        assert leaves == []
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored == foo

    def test_two_different_subclasses_independent(self):
        class Foo(DataClass):
            x: float

        class Bar(DataClass):
            y: int = static_field(default=0)

        foo = Foo(x=1.0)
        bar = Bar()
        assert foo != bar  # different types
        assert len(jax.tree_util.tree_leaves(foo)) == 1
        assert len(jax.tree_util.tree_leaves(bar)) == 0


# ---------------------------------------------------------------------------
# _parse_operations (internal parser)
# ---------------------------------------------------------------------------


class TestParseOperations:
    def test_empty_string_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(ValueError, match="Empty string"):
            foo._parse_operations("")

    def test_simple_attribute(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("x")
        assert ops == [("x", "attribute")]

    def test_integer_index(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("[0]")
        assert ops == [(0, "index")]

    def test_negative_index(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("[-1]")
        assert ops == [(-1, "index")]

    def test_string_key(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("['mykey']")
        assert ops == [("mykey", "key")]

    def test_string_key_with_spaces(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("['key with spaces']")
        assert ops == [("key with spaces", "key")]

    def test_chained_attributes(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("a->b->c")
        assert ops == [("a", "attribute"), ("b", "attribute"), ("c", "attribute")]

    def test_attribute_then_index(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("items->[0]")
        assert ops == [("items", "attribute"), (0, "index")]

    def test_attribute_then_key(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("data->['name']")
        assert ops == [("data", "attribute"), ("name", "key")]

    def test_complex_chain(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("a->b->[0]->['name']")
        assert ops == [
            ("a", "attribute"),
            ("b", "attribute"),
            (0, "index"),
            ("name", "key"),
        ]

    def test_unclosed_bracket_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(ValueError, match="Unclosed bracket"):
            foo._parse_operations("[0")

    def test_trailing_arrow_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(ValueError, match="ends with '->'"):
            foo._parse_operations("a->")

    def test_invalid_bracket_content_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(ValueError, match="Invalid bracket content"):
            foo._parse_operations("[abc]")

    def test_string_key_with_single_quote_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(ValueError):
            foo._parse_operations("['key'embedded']")

    def test_string_key_with_bracket_in_content_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(ValueError):
            foo._parse_operations("['key[0]']")

    def test_missing_separator_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        # After a bracket op, the parser expects "->"; a bare letter instead raises
        with pytest.raises(ValueError, match="Expected '->'"):
            foo._parse_operations("[0]b")

    def test_large_positive_index(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("[999]")
        assert ops == [(999, "index")]

    def test_single_char_attribute(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        ops = foo._parse_operations("x")
        assert ops == [("x", "attribute")]

    def test_underscore_attribute(self):
        class Foo(DataClass):
            _x: float = field(default=1.0, init=True)

        foo = Foo(_x=1.0)
        ops = foo._parse_operations("_x")
        assert ops == [("_x", "attribute")]


# ---------------------------------------------------------------------------
# updated_copy method
# ---------------------------------------------------------------------------


class TestUpdatedCopy:
    def test_single_field_update(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        updated = foo.updated_copy(x=9.0)
        assert updated.x == 9.0
        assert updated.y == 2.0

    def test_multiple_fields_update(self):
        class Foo(DataClass):
            x: float
            y: float
            z: float

        foo = Foo(x=1.0, y=2.0, z=3.0)
        updated = foo.updated_copy(x=10.0, z=30.0)
        assert updated.x == 10.0
        assert updated.y == 2.0
        assert updated.z == 30.0

    def test_original_unchanged(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        _ = foo.updated_copy(x=99.0)
        assert foo.x == 1.0

    def test_returns_same_type(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        updated = foo.updated_copy(x=2.0)
        assert type(updated) is Foo

    def test_no_kwargs_returns_equal_copy(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        copy = foo.updated_copy()
        assert copy == foo
        assert copy is not foo

    def test_update_static_field(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=5)

        foo = Foo(x=1.0, n=5)
        updated = foo.updated_copy(n=10)
        assert updated.n == 10
        assert updated.x == 1.0

    def test_update_dynamic_field(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=1)

        foo = Foo(x=1.0)
        updated = foo.updated_copy(x=42.0)
        assert updated.x == 42.0
        assert updated.n == 1

    def test_nonexistent_field_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(TypeError):
            foo.updated_copy(nonexistent=99.0)

    def test_update_with_none(self):
        class Foo(DataClass):
            x: float
            extra: object = field(default=None)

        foo = Foo(x=1.0, extra="something")
        updated = foo.updated_copy(extra=None)
        assert updated.extra is None
        assert updated.x == 1.0

    def test_update_with_jax_array(self):
        class Foo(DataClass):
            x: jax.Array

        foo = Foo(x=jnp.array(1.0))
        updated = foo.updated_copy(x=jnp.array(99.0))
        assert float(updated.x) == pytest.approx(99.0)
        assert float(foo.x) == pytest.approx(1.0)

    def test_update_nested_dataclass_field(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner
            bias: float

        outer = Outer(inner=Inner(w=1.0), bias=0.5)
        new_inner = Inner(w=99.0)
        updated = outer.updated_copy(inner=new_inner)
        assert updated.inner.w == 99.0
        assert updated.bias == 0.5
        assert outer.inner.w == 1.0  # original unchanged

    def test_updated_copy_is_valid_pytree(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        updated = foo.updated_copy(x=10.0)
        leaves = jax.tree_util.tree_leaves(updated)
        assert 10.0 in leaves
        assert 2.0 in leaves

    def test_updated_copy_works_under_jit(self):
        class Foo(DataClass):
            x: jax.Array
            n: int = static_field(default=1)

        @jax.jit
        def f(foo):
            return foo.updated_copy(x=foo.x * 2)

        foo = Foo(x=jnp.array(3.0))
        result = f(foo)
        assert float(result.x) == pytest.approx(6.0)
        assert result.n == 1

    def test_all_fields_updated(self):
        class Foo(DataClass):
            a: float
            b: float
            c: float

        foo = Foo(a=1.0, b=2.0, c=3.0)
        updated = foo.updated_copy(a=10.0, b=20.0, c=30.0)
        assert updated.a == 10.0
        assert updated.b == 20.0
        assert updated.c == 30.0

    def test_update_private_field_raises(self):
        # dataclasses.replace() disallows init=False fields; private_field sets init=False
        class Foo(DataClass):
            x: float
            _cache: float = private_field(default=0.0)

        foo = Foo(x=1.0)
        with pytest.raises((TypeError, ValueError)):
            foo.updated_copy(_cache=5.0)


# ---------------------------------------------------------------------------
# aset method — top-level attribute
# ---------------------------------------------------------------------------


class TestAsetTopLevel:
    def test_simple_attribute_update(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        updated = foo.aset("x", 9.0)
        assert updated.x == 9.0

    def test_original_unchanged(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        _ = foo.aset("x", 99.0)
        assert foo.x == 1.0

    def test_returns_same_type(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        updated = foo.aset("x", 2.0)
        assert type(updated) is Foo

    def test_multiple_field_class_preserves_others(self):
        class Foo(DataClass):
            x: float
            y: float
            z: float

        foo = Foo(x=1.0, y=2.0, z=3.0)
        updated = foo.aset("y", 20.0)
        assert updated.x == 1.0
        assert updated.y == 20.0
        assert updated.z == 3.0

    def test_nonexistent_attribute_raises(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises(Exception, match="does not exist"):
            foo.aset("nonexistent", 99.0)

    def test_nonexistent_attribute_create_new_ok_true_still_raises(self):
        # create_new_ok=True only skips the existence check on the walk-down;
        # dataclasses.replace will still reject unknown fields.
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        with pytest.raises((TypeError, Exception)):
            foo.aset("nonexistent", 99.0, create_new_ok=True)

    def test_update_static_field(self):
        class Foo(DataClass):
            x: float
            n: int = static_field(default=5)

        foo = Foo(x=1.0)
        updated = foo.aset("n", 99)
        assert updated.n == 99
        assert updated.x == 1.0

    def test_update_to_none(self):
        class Foo(DataClass):
            x: float
            extra: object = field(default=None)

        foo = Foo(x=1.0, extra="hello")
        updated = foo.aset("extra", None)
        assert updated.extra is None

    def test_update_with_jax_array(self):
        class Foo(DataClass):
            x: jax.Array

        foo = Foo(x=jnp.array(1.0))
        updated = foo.aset("x", jnp.array(42.0))
        assert float(updated.x) == pytest.approx(42.0)

    def test_update_private_field_raises(self):
        # dataclasses.replace() disallows init=False fields; private_field sets init=False
        class Foo(DataClass):
            x: float
            _cache: float = private_field(default=0.0)

        foo = Foo(x=1.0)
        with pytest.raises((TypeError, ValueError)):
            foo.aset("_cache", 5.0)

    def test_single_field_class(self):
        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        assert foo.aset("x", 2.0).x == 2.0

    def test_result_is_valid_pytree(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        updated = foo.aset("x", 10.0)
        leaves = jax.tree_util.tree_leaves(updated)
        assert 10.0 in leaves
        assert 2.0 in leaves


# ---------------------------------------------------------------------------
# aset method — nested DataClass attributes
# ---------------------------------------------------------------------------


class TestAsetNestedDataClass:
    def test_nested_attribute_update(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner
            bias: float

        outer = Outer(inner=Inner(w=1.0), bias=0.5)
        updated = outer.aset("inner->w", 99.0)
        assert updated.inner.w == 99.0
        assert updated.bias == 0.5

    def test_nested_original_unchanged(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(w=1.0))
        _ = outer.aset("inner->w", 99.0)
        assert outer.inner.w == 1.0

    def test_deeply_nested(self):
        class A(DataClass):
            val: float

        class B(DataClass):
            a: A
            x: float

        class C(DataClass):
            b: B
            y: float

        obj = C(b=B(a=A(val=1.0), x=2.0), y=3.0)
        updated = obj.aset("b->a->val", 99.0)
        assert updated.b.a.val == 99.0
        assert updated.b.x == 2.0
        assert updated.y == 3.0

    def test_deeply_nested_intermediate_unchanged(self):
        class A(DataClass):
            val: float

        class B(DataClass):
            a: A

        class C(DataClass):
            b: B

        obj = C(b=B(a=A(val=1.0)))
        _ = obj.aset("b->a->val", 99.0)
        assert obj.b.a.val == 1.0

    def test_nested_returns_same_outer_type(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(w=1.0))
        updated = outer.aset("inner->w", 2.0)
        assert type(updated) is Outer

    def test_nested_nonexistent_intermediate_raises(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(w=1.0))
        with pytest.raises(Exception):
            outer.aset("nonexistent->w", 1.0)

    def test_nested_nonexistent_leaf_raises(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(w=1.0))
        with pytest.raises(Exception):
            outer.aset("inner->nonexistent", 1.0)

    def test_replace_entire_inner(self):
        class Inner(DataClass):
            w: float

        class Outer(DataClass):
            inner: Inner
            bias: float

        outer = Outer(inner=Inner(w=1.0), bias=0.5)
        updated = outer.aset("inner", Inner(w=99.0))
        assert updated.inner.w == 99.0
        assert updated.bias == 0.5

    def test_nested_static_field_update(self):
        class Inner(DataClass):
            w: float
            n: int = static_field(default=1)

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(w=1.0))
        updated = outer.aset("inner->n", 7)
        assert updated.inner.n == 7
        assert updated.inner.w == 1.0

    def test_nested_with_jax_array(self):
        class Inner(DataClass):
            w: jax.Array

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(w=jnp.array(1.0)))
        updated = outer.aset("inner->w", jnp.array(42.0))
        assert float(updated.inner.w) == pytest.approx(42.0)

    def test_three_level_sibling_fields_preserved(self):
        class A(DataClass):
            val: float
            extra: float

        class B(DataClass):
            a: A
            b_val: float

        class C(DataClass):
            b: B
            c_val: float

        obj = C(b=B(a=A(val=1.0, extra=5.0), b_val=2.0), c_val=3.0)
        updated = obj.aset("b->a->val", 99.0)
        assert updated.b.a.val == 99.0
        assert updated.b.a.extra == 5.0
        assert updated.b.b_val == 2.0
        assert updated.c_val == 3.0


# ---------------------------------------------------------------------------
# aset method — list index operations
# ---------------------------------------------------------------------------


class TestAsetListIndex:
    def test_update_list_element(self):
        class Foo(DataClass):
            items: list

        foo = Foo(items=[1.0, 2.0, 3.0])
        updated = foo.aset("items->[0]", 99.0)
        assert updated.items[0] == 99.0
        assert updated.items[1] == 2.0
        assert updated.items[2] == 3.0

    def test_update_list_last_element(self):
        class Foo(DataClass):
            items: list

        foo = Foo(items=[1.0, 2.0, 3.0])
        updated = foo.aset("items->[2]", 99.0)
        assert updated.items[2] == 99.0
        assert updated.items[0] == 1.0

    def test_update_list_negative_index(self):
        class Foo(DataClass):
            items: list

        foo = Foo(items=[1.0, 2.0, 3.0])
        updated = foo.aset("items->[-1]", 99.0)
        assert updated.items[-1] == 99.0
        assert updated.items[0] == 1.0

    def test_original_list_unchanged(self):
        class Foo(DataClass):
            items: list

        original_list = [1.0, 2.0, 3.0]
        foo = Foo(items=original_list)
        _ = foo.aset("items->[0]", 99.0)
        assert foo.items[0] == 1.0
        assert original_list[0] == 1.0

    def test_update_nested_list_element(self):
        class Inner(DataClass):
            vals: list

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(vals=[10.0, 20.0, 30.0]))
        updated = outer.aset("inner->vals->[1]", 99.0)
        assert updated.inner.vals[1] == 99.0
        assert updated.inner.vals[0] == 10.0
        assert updated.inner.vals[2] == 30.0

    def test_list_element_is_dataclass(self):
        class Item(DataClass):
            val: float

        class Foo(DataClass):
            items: list

        foo = Foo(items=[Item(val=1.0), Item(val=2.0)])
        new_item = Item(val=99.0)
        updated = foo.aset("items->[0]", new_item)
        assert updated.items[0].val == 99.0
        assert updated.items[1].val == 2.0

    def test_object_without_getitem_raises(self):
        class Inner(DataClass):
            x: float

        class Foo(DataClass):
            inner: Inner

        foo = Foo(inner=Inner(x=1.0))
        with pytest.raises(Exception):
            foo.aset("inner->[0]", 99.0)


# ---------------------------------------------------------------------------
# aset method — dict key operations
# ---------------------------------------------------------------------------


class TestAsetDictKey:
    def test_update_dict_value(self):
        class Foo(DataClass):
            data: dict

        foo = Foo(data={"a": 1.0, "b": 2.0})
        updated = foo.aset("data->['a']", 99.0)
        assert updated.data["a"] == 99.0
        assert updated.data["b"] == 2.0

    def test_original_dict_unchanged(self):
        class Foo(DataClass):
            data: dict

        original = {"a": 1.0}
        foo = Foo(data=original)
        _ = foo.aset("data->['a']", 99.0)
        assert foo.data["a"] == 1.0
        assert original["a"] == 1.0

    def test_nonexistent_key_raises(self):
        class Foo(DataClass):
            data: dict

        foo = Foo(data={"a": 1.0})
        with pytest.raises(Exception, match="does not exist|Key"):
            foo.aset("data->['missing']", 99.0)

    def test_nonexistent_key_create_new_ok(self):
        class Foo(DataClass):
            data: dict

        foo = Foo(data={"a": 1.0})
        updated = foo.aset("data->['new_key']", 42.0, create_new_ok=True)
        assert updated.data["new_key"] == 42.0
        assert updated.data["a"] == 1.0

    def test_nested_dict_update(self):
        class Foo(DataClass):
            data: dict

        foo = Foo(data={"x": {"inner": 1.0, "other": 2.0}})
        updated = foo.aset("data->['x']->['inner']", 99.0)
        assert updated.data["x"]["inner"] == 99.0
        assert updated.data["x"]["other"] == 2.0

    def test_dict_value_is_dataclass(self):
        class Item(DataClass):
            val: float

        class Foo(DataClass):
            data: dict

        foo = Foo(data={"item": Item(val=1.0)})
        new_item = Item(val=99.0)
        updated = foo.aset("data->['item']", new_item)
        assert updated.data["item"].val == 99.0

    def test_dict_value_nested_in_dataclass(self):
        class Inner(DataClass):
            val: float

        class Foo(DataClass):
            data: dict

        foo = Foo(data={"key": Inner(val=1.0)})
        updated = foo.aset("data->['key']->val", 99.0)
        assert updated.data["key"].val == 99.0

    def test_create_new_key_in_nested_dict(self):
        class Foo(DataClass):
            outer: dict

        foo = Foo(outer={"inner": {"existing": 1.0}})
        updated = foo.aset("outer->['inner']->['new']", 99.0, create_new_ok=True)
        assert updated.outer["inner"]["new"] == 99.0
        assert updated.outer["inner"]["existing"] == 1.0

    def test_update_preserves_other_keys(self):
        class Foo(DataClass):
            data: dict

        foo = Foo(data={"a": 1.0, "b": 2.0, "c": 3.0})
        updated = foo.aset("data->['b']", 99.0)
        assert updated.data["a"] == 1.0
        assert updated.data["b"] == 99.0
        assert updated.data["c"] == 3.0


# ---------------------------------------------------------------------------
# aset method — mixed path operations
# ---------------------------------------------------------------------------


class TestAsetMixedPaths:
    def test_attr_then_index_then_key(self):
        class Foo(DataClass):
            items: list

        foo = Foo(items=[{"name": "alice"}, {"name": "bob"}])
        updated = foo.aset("items->[0]->['name']", "charlie")
        assert updated.items[0]["name"] == "charlie"
        assert updated.items[1]["name"] == "bob"

    def test_attr_then_key_then_index(self):
        class Foo(DataClass):
            data: dict

        foo = Foo(data={"vals": [1.0, 2.0, 3.0]})
        updated = foo.aset("data->['vals']->[2]", 99.0)
        assert updated.data["vals"][2] == 99.0
        assert updated.data["vals"][0] == 1.0

    def test_dataclass_inside_list(self):
        class Item(DataClass):
            val: float
            name: str = static_field(default="item")

        class Foo(DataClass):
            items: list

        foo = Foo(items=[Item(val=1.0), Item(val=2.0)])
        updated = foo.aset("items->[1]->val", 99.0)
        assert updated.items[1].val == 99.0
        assert updated.items[0].val == 1.0

    def test_dataclass_inside_dict(self):
        class Item(DataClass):
            val: float

        class Foo(DataClass):
            registry: dict

        foo = Foo(registry={"first": Item(val=1.0), "second": Item(val=2.0)})
        updated = foo.aset("registry->['first']->val", 99.0)
        assert updated.registry["first"].val == 99.0
        assert updated.registry["second"].val == 2.0

    def test_outer_dataclass_attr_then_inner_list(self):
        class Inner(DataClass):
            vals: list

        class Outer(DataClass):
            inner: Inner
            bias: float

        outer = Outer(inner=Inner(vals=[1.0, 2.0]), bias=0.5)
        updated = outer.aset("inner->vals->[0]", 99.0)
        assert updated.inner.vals[0] == 99.0
        assert updated.inner.vals[1] == 2.0
        assert updated.bias == 0.5

    def test_chain_returns_self_type(self):
        class Inner(DataClass):
            data: dict

        class Outer(DataClass):
            inner: Inner

        outer = Outer(inner=Inner(data={"x": 1.0}))
        updated = outer.aset("inner->data->['x']", 99.0)
        assert type(updated) is Outer

    def test_aset_chained_calls(self):
        class Foo(DataClass):
            x: float
            y: float

        foo = Foo(x=1.0, y=2.0)
        updated = foo.aset("x", 10.0).aset("y", 20.0)
        assert updated.x == 10.0
        assert updated.y == 20.0

    def test_aset_with_jax_array_in_list(self):
        class Foo(DataClass):
            items: list

        foo = Foo(items=[jnp.array(1.0), jnp.array(2.0)])
        updated = foo.aset("items->[0]", jnp.array(99.0))
        assert float(updated.items[0]) == pytest.approx(99.0)
        assert float(updated.items[1]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# aset_inplace
# ---------------------------------------------------------------------------


class TestAsetInplace:
    def test_simple_top_level_in_post_init(self):
        class Foo(DataClass):
            x: float
            doubled: float = private_field()

            def __post_init__(self) -> None:
                self.aset_inplace("doubled", self.x * 2)

        foo = Foo(x=3.0)  # ty:ignore[missing-argument]
        assert foo.doubled == 6.0

    def test_nested_path(self):
        class Inner(DataClass):
            b: float

        class Outer(DataClass):
            a: Inner
            b_cache: float = private_field()

            def __post_init__(self) -> None:
                self.aset_inplace("b_cache", self.a.b + 1.0)

        outer = Outer(a=Inner(b=5.0))  # ty:ignore[missing-argument]
        assert outer.b_cache == 6.0

    def test_list_index(self):
        class Foo(DataClass):
            items: list

            def __post_init__(self) -> None:
                # mutate the list element in-place (list is mutable)
                self.aset_inplace("items->[0]", 99.0)

        foo = Foo(items=[1.0, 2.0])
        assert foo.items[0] == 99.0
        assert foo.items[1] == 2.0

    def test_dict_key(self):
        class Foo(DataClass):
            data: dict

            def __post_init__(self) -> None:
                self.aset_inplace("data->['x']", 42.0)

        foo = Foo(data={"x": 0.0, "y": 1.0})
        assert foo.data["x"] == 42.0
        assert foo.data["y"] == 1.0

    def test_mutates_original_unlike_aset(self):
        """aset_inplace mutates self; aset returns a new object."""

        class Foo(DataClass):
            x: float

        foo = Foo(x=1.0)
        # aset: returns new object, original unchanged (frozen)
        new_foo = foo.aset("x", 2.0)
        assert foo.x == 1.0
        assert new_foo.x == 2.0

        # aset_inplace: mutates in place, returns None
        result = foo.aset_inplace("x", 3.0)
        assert result is None
        assert foo.x == 3.0

    def test_private_field_scenario(self):
        """Typical __post_init__ pattern: derive a private field from init fields."""

        class Model(DataClass):
            weights: list
            n_params: int = private_field()

            def __post_init__(self) -> None:
                self.aset_inplace("n_params", len(self.weights))

        m = Model(weights=[1.0, 2.0, 3.0])  # ty:ignore[missing-argument]
        assert m.n_params == 3
