"""Tests for JAX transform compatibility: jit, vmap, grad, and others."""

import jax
import jax.numpy as jnp
import pytest

from drinx import dataclass, static_field


# ---------------------------------------------------------------------------
# jax.jit
# ---------------------------------------------------------------------------


class TestJit:
    def test_jit_basic(self):
        @dataclass
        class Foo:
            x: jax.Array

        @jax.jit
        def f(foo):
            return foo.x * 2

        foo = Foo(x=jnp.array(3.0))
        result = f(foo)
        assert float(result) == pytest.approx(6.0)

    def test_jit_multiple_dynamic_fields(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        @jax.jit
        def f(foo):
            return foo.x + foo.y

        foo = Foo(x=jnp.array(1.0), y=jnp.array(2.0))
        assert float(f(foo)) == pytest.approx(3.0)

    def test_jit_with_static_field(self):
        @dataclass
        class Foo:
            x: jax.Array
            n: int = static_field(default=3)

        @jax.jit
        def f(foo):
            return foo.x * foo.n

        foo = Foo(x=jnp.array(2.0))
        assert float(f(foo)) == pytest.approx(6.0)

    def test_jit_returns_same_type(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        @jax.jit
        def f(foo):
            return Foo(x=foo.x * 2, y=foo.y + 1)

        result = f(Foo(x=jnp.array(1.0), y=jnp.array(2.0)))
        assert isinstance(result, Foo)
        assert float(result.x) == pytest.approx(2.0)
        assert float(result.y) == pytest.approx(3.0)

    def test_jit_static_field_triggers_recompile_on_change(self):
        call_count = 0

        @dataclass
        class Foo:
            x: jax.Array
            n: int = static_field(default=1)

        @jax.jit
        def f(foo):
            nonlocal call_count
            call_count += 1
            return foo.x * foo.n

        f(Foo(x=jnp.array(1.0), n=2))
        f(Foo(x=jnp.array(2.0), n=2))  # same static, no recompile
        count_after_same_static = call_count

        f(Foo(x=jnp.array(1.0), n=3))  # different static, recompile
        assert call_count > count_after_same_static

    def test_jit_all_static_fields(self):
        @dataclass
        class Foo:
            n: int = static_field(default=2)
            m: int = static_field(default=3)

        @jax.jit
        def f(foo):
            return jnp.array(foo.n + foo.m, dtype=jnp.float32)

        foo = Foo()
        assert float(f(foo)) == pytest.approx(5.0)

    def test_jit_no_fields(self):
        @dataclass
        class Empty:
            pass

        @jax.jit
        def f(obj):
            return jnp.array(1.0)

        result = f(Empty())
        assert float(result) == pytest.approx(1.0)

    def test_jit_vectorized_input(self):
        @dataclass
        class Foo:
            x: jax.Array

        @jax.jit
        def f(foo):
            return foo.x**2

        foo = Foo(x=jnp.array([1.0, 2.0, 3.0]))
        result = f(foo)
        assert result.shape == (3,)
        assert float(result[0]) == pytest.approx(1.0)
        assert float(result[1]) == pytest.approx(4.0)
        assert float(result[2]) == pytest.approx(9.0)

    def test_jit_nested_dataclasses(self):
        @dataclass
        class Inner:
            w: jax.Array

        @dataclass
        class Outer:
            inner: Inner
            bias: jax.Array

        @jax.jit
        def f(outer):
            return outer.inner.w + outer.bias

        obj = Outer(inner=Inner(w=jnp.array(1.0)), bias=jnp.array(0.5))
        assert float(f(obj)) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# jax.grad
# ---------------------------------------------------------------------------


class TestGrad:
    def test_grad_through_dynamic_field(self):
        @dataclass
        class Foo:
            x: jax.Array

        def loss(foo):
            return foo.x**2

        foo = Foo(x=jnp.array(3.0))
        grads = jax.grad(loss)(foo)
        assert float(grads.x) == pytest.approx(6.0)

    def test_grad_multiple_dynamic_fields(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        def loss(foo):
            return foo.x**2 + foo.y**3

        foo = Foo(x=jnp.array(2.0), y=jnp.array(3.0))
        grads = jax.grad(loss)(foo)
        assert float(grads.x) == pytest.approx(4.0)  # d/dx x^2 = 2x
        assert float(grads.y) == pytest.approx(27.0)  # d/dy y^3 = 3y^2

    def test_grad_with_static_field(self):
        @dataclass
        class Foo:
            x: jax.Array
            scale: float = static_field(default=2.0)

        def loss(foo):
            return foo.x**2 * foo.scale

        foo = Foo(x=jnp.array(3.0))
        grads = jax.grad(loss)(foo)
        assert float(grads.x) == pytest.approx(12.0)  # d/dx 2x^2 = 4x at x=3

    def test_grad_returns_same_type(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        def loss(foo):
            return foo.x + foo.y

        foo = Foo(x=jnp.array(1.0), y=jnp.array(2.0))
        grads = jax.grad(loss)(foo)
        assert isinstance(grads, Foo)

    def test_grad_jit_composed(self):
        @dataclass
        class Foo:
            x: jax.Array

        def loss(foo):
            return foo.x**2

        foo = Foo(x=jnp.array(4.0))
        grad_fn = jax.jit(jax.grad(loss))
        grads = grad_fn(foo)
        assert float(grads.x) == pytest.approx(8.0)

    def test_value_and_grad(self):
        @dataclass
        class Foo:
            x: jax.Array

        def loss(foo):
            return foo.x**2

        foo = Foo(x=jnp.array(3.0))
        value, grads = jax.value_and_grad(loss)(foo)
        assert float(value) == pytest.approx(9.0)
        assert float(grads.x) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# jax.vmap
# ---------------------------------------------------------------------------


class TestVmap:
    def test_vmap_basic(self):
        @dataclass
        class Foo:
            x: jax.Array

        @jax.vmap
        def f(foo):
            return foo.x * 2

        foo = Foo(x=jnp.array([1.0, 2.0, 3.0]))
        result = f(foo)
        assert result.shape == (3,)
        assert float(result[0]) == pytest.approx(2.0)
        assert float(result[2]) == pytest.approx(6.0)

    def test_vmap_multiple_fields(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        @jax.vmap
        def f(foo):
            return foo.x + foo.y

        foo = Foo(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]))
        result = f(foo)
        assert float(result[0]) == pytest.approx(4.0)
        assert float(result[1]) == pytest.approx(6.0)

    def test_vmap_returns_batched_dataclass(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        @jax.vmap
        def f(foo):
            return Foo(x=foo.x * 2, y=foo.y + 1)

        foo = Foo(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]))
        result = f(foo)
        assert isinstance(result, Foo)
        assert result.x.shape == (2,)
        assert float(result.x[0]) == pytest.approx(2.0)
        assert float(result.y[1]) == pytest.approx(5.0)

    def test_vmap_with_static_field(self):
        @dataclass
        class Foo:
            x: jax.Array
            scale: float = static_field(default=3.0)

        @jax.vmap
        def f(foo):
            return foo.x * foo.scale

        foo = Foo(x=jnp.array([1.0, 2.0, 4.0]))
        result = f(foo)
        assert float(result[0]) == pytest.approx(3.0)
        assert float(result[1]) == pytest.approx(6.0)
        assert float(result[2]) == pytest.approx(12.0)

    def test_vmap_nested(self):
        @dataclass
        class Inner:
            w: jax.Array

        @dataclass
        class Outer:
            inner: Inner
            bias: jax.Array

        @jax.vmap
        def f(outer):
            return outer.inner.w + outer.bias

        obj = Outer(
            inner=Inner(w=jnp.array([1.0, 2.0])),
            bias=jnp.array([0.5, 0.5]),
        )
        result = f(obj)
        assert float(result[0]) == pytest.approx(1.5)
        assert float(result[1]) == pytest.approx(2.5)

    def test_vmap_jit_composed(self):
        @dataclass
        class Foo:
            x: jax.Array

        f = jax.jit(jax.vmap(lambda foo: foo.x**2))
        foo = Foo(x=jnp.array([2.0, 3.0]))
        result = f(foo)
        assert float(result[0]) == pytest.approx(4.0)
        assert float(result[1]) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# jax.lax.scan
# ---------------------------------------------------------------------------


class TestScan:
    def test_scan_as_carry(self):
        @dataclass
        class State:
            x: jax.Array

        def step(carry, _):
            new_x = carry.x + 1.0
            return State(x=new_x), new_x

        init = State(x=jnp.array(0.0))
        final, outputs = jax.lax.scan(step, init, None, length=5)
        assert float(final.x) == pytest.approx(5.0)
        assert outputs.shape == (5,)

    def test_scan_with_static_field(self):
        @dataclass
        class State:
            x: jax.Array
            step_size: float = static_field(default=2.0)

        def step(carry, _):
            new_x = carry.x + carry.step_size
            return State(x=new_x, step_size=carry.step_size), new_x

        init = State(x=jnp.array(0.0))
        final, _ = jax.lax.scan(step, init, None, length=3)
        assert float(final.x) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


class TestTreeUtils:
    def test_tree_map_preserves_structure(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        foo = Foo(x=jnp.array(1.0), y=jnp.array(2.0))
        doubled = jax.tree_util.tree_map(lambda v: v * 2, foo)
        assert isinstance(doubled, Foo)
        assert float(doubled.x) == pytest.approx(2.0)
        assert float(doubled.y) == pytest.approx(4.0)

    def test_tree_map_two_pytrees(self):
        @dataclass
        class Foo:
            x: jax.Array

        a = Foo(x=jnp.array(1.0))
        b = Foo(x=jnp.array(3.0))
        result = jax.tree_util.tree_map(lambda u, v: u + v, a, b)
        assert float(result.x) == pytest.approx(4.0)

    def test_tree_leaves_order_matches_fields(self):
        @dataclass
        class Foo:
            a: float
            b: float
            c: float

        foo = Foo(a=1.0, b=2.0, c=3.0)
        leaves = jax.tree_util.tree_leaves(foo)
        # dynamic fields appear in definition order
        assert leaves == [1.0, 2.0, 3.0]

    def test_tree_leaves_with_jnp_arrays(self):
        @dataclass
        class Foo:
            x: jax.Array
            y: jax.Array

        foo = Foo(x=jnp.zeros((3,)), y=jnp.ones((2,)))
        leaves = jax.tree_util.tree_leaves(foo)
        assert len(leaves) == 2
        assert leaves[0].shape == (3,)
        assert leaves[1].shape == (2,)
