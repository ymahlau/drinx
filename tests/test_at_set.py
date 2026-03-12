"""Tests for the .at[].set() fluent update API on DataClass."""

import jax
import jax.numpy as jnp
import pytest

import drinx
from drinx import DataClass, static_field


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class Simple(DataClass):
    a: jax.Array
    b: jax.Array


class WithStatic(DataClass):
    weights: jax.Array
    lr: float = static_field(default=1e-3)


class Nested(DataClass):
    inner: Simple
    c: jax.Array


# ---------------------------------------------------------------------------
# Basic attribute update
# ---------------------------------------------------------------------------


def test_at_set_basic_field():
    tree = Simple(a=jnp.array(1.0), b=jnp.array(2.0))
    result = tree.at["a"].set(jnp.array(99.0))
    assert float(result.a) == 99.0
    assert float(result.b) == 2.0  # unchanged


def test_at_set_returns_same_type():
    tree = Simple(a=jnp.array(1.0), b=jnp.array(2.0))
    result = tree.at["b"].set(jnp.array(42.0))
    assert type(result) is Simple


# ---------------------------------------------------------------------------
# Attribute + integer index
# ---------------------------------------------------------------------------


def test_at_set_field_then_index():
    """tree.at['field'][0].set(value) — attribute then integer index into list."""

    @drinx.dataclass
    class WithList(DataClass):
        items: list

    tree = WithList(items=[10, 20, 30])
    result = tree.at["items"][1].set(99)
    assert result.items == [10, 99, 30]
    assert tree.items == [10, 20, 30]  # original unchanged


# ---------------------------------------------------------------------------
# Chained updates
# ---------------------------------------------------------------------------


def test_at_set_chained():
    tree = Simple(a=jnp.array(1.0), b=jnp.array(2.0))
    result = tree.at["a"].set(jnp.array(10.0)).at["b"].set(jnp.array(20.0))
    assert float(result.a) == 10.0
    assert float(result.b) == 20.0


# ---------------------------------------------------------------------------
# Mask updates
# ---------------------------------------------------------------------------


def test_at_set_mask_scalar():
    tree = Simple(a=jnp.array([1.0, 6.0, 3.0]), b=jnp.array([7.0, 2.0, 8.0]))
    mask = jax.tree.map(lambda x: x > 5, tree)
    result = tree.at[mask].set(0.0)
    assert jnp.allclose(result.a, jnp.array([1.0, 0.0, 3.0]))
    assert jnp.allclose(result.b, jnp.array([0.0, 2.0, 0.0]))


def test_at_set_mask_tree_value():
    """Mask update where value is a full-tree of the same type."""
    tree = Simple(a=jnp.array([1.0, 6.0]), b=jnp.array([7.0, 2.0]))
    mask = jax.tree.map(lambda x: x > 5, tree)
    fill = Simple(a=jnp.array([-1.0, -1.0]), b=jnp.array([-1.0, -1.0]))
    result = tree.at[mask].set(fill)
    assert jnp.allclose(result.a, jnp.array([1.0, -1.0]))
    assert jnp.allclose(result.b, jnp.array([-1.0, 2.0]))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_at_set_invalid_field_name():
    tree = Simple(a=jnp.array(1.0), b=jnp.array(2.0))
    with pytest.raises(Exception, match="does not exist"):
        tree.at["nonexistent"].set(1.0)


def test_at_set_wrong_mask_type():
    """A mask that is NOT the same DataClass type should raise TypeError."""
    tree = Simple(a=jnp.array(1.0), b=jnp.array(2.0))
    with pytest.raises(TypeError, match="Unsupported key type"):
        tree.at[object()].set(1.0)


def test_at_set_wrong_key_type_in_path():
    """Non-str/int key in a multi-step path should raise TypeError."""
    tree = Simple(a=jnp.array(1.0), b=jnp.array(2.0))
    with pytest.raises(TypeError, match="Unsupported key type"):
        tree.at["a"][3.14].set(1.0)  # float key — not supported
