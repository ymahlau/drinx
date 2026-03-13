"""Tests that execute the exact code snippets from README.md to ensure they work."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import drinx


# ---------------------------------------------------------------------------
# Decorator style
# ---------------------------------------------------------------------------


class TestDecoratorStyleExample:
    def test_basic_decorator_and_tree_map(self):
        @drinx.dataclass
        class Params:
            weights: jax.Array
            bias: jax.Array

        params = Params(weights=jnp.ones((3,)), bias=jnp.zeros((3,)))

        # Works transparently with JAX transforms
        doubled = jax.tree_util.tree_map(lambda x: x * 2, params)

        assert isinstance(doubled, Params)
        assert doubled.weights.shape == (3,)
        assert doubled.bias.shape == (3,)
        np.testing.assert_allclose(doubled.weights, jnp.full((3,), 2.0))
        np.testing.assert_allclose(doubled.bias, jnp.zeros((3,)))


# ---------------------------------------------------------------------------
# Static fields
# ---------------------------------------------------------------------------


class TestStaticFieldsExample:
    def test_static_field_jit(self):
        @drinx.dataclass
        class Model:
            weights: jax.Array
            hidden_size: int = drinx.static_field(default=128)

        @jax.jit
        def forward(model, x):
            # hidden_size is a compile-time constant; weights are traced
            return model.weights[: model.hidden_size] @ x

        model = Model(weights=jnp.ones((128, 32)))
        x = jnp.ones((32,))
        result = forward(model, x)
        assert result.shape == (128,)
        np.testing.assert_allclose(result, jnp.full((128,), 32.0))


# ---------------------------------------------------------------------------
# Inheritance style
# ---------------------------------------------------------------------------


class TestInheritanceStyleExample:
    def test_basic_inheritance(self):
        class Model(drinx.DataClass):
            weights: jax.Array
            learning_rate: float = drinx.static_field(default=1e-3)

        model = Model(weights=jnp.ones((10,)))
        assert model.weights.shape == (10,)
        assert model.learning_rate == pytest.approx(1e-3)

    def test_inheritance_with_kw_only_and_order(self):
        class Config(drinx.DataClass, kw_only=True, order=True):
            hidden_size: int = drinx.static_field(default=128)
            num_layers: int = drinx.static_field(default=4)

        cfg = Config()
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 4

    def test_decorator_plus_inheritance_kw_only_order(self):
        # Recommended way: typechecker recognises kw_only correctly
        @drinx.dataclass(kw_only=True, order=True)
        class Config(drinx.DataClass):
            hidden_size: int = drinx.static_field(default=128)
            num_layers: int = drinx.static_field(default=4)

        cfg = Config()
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 4
        # order=True means comparison works
        assert Config(hidden_size=64) < Config(hidden_size=128)


# ---------------------------------------------------------------------------
# Functional updates with aset
# ---------------------------------------------------------------------------


class TestAsetExample:
    def test_aset_top_level_field(self):
        class Inner(drinx.DataClass):
            w: jax.Array

        @drinx.dataclass
        class Outer(drinx.DataClass):
            inner: Inner
            bias: jax.Array

        outer = Outer(inner=Inner(w=jnp.ones((3,))), bias=jnp.zeros((3,)))

        # Update a top-level field
        outer2 = outer.aset("bias", jnp.ones((3,)))

        assert isinstance(outer2, Outer)
        np.testing.assert_allclose(outer2.bias, jnp.ones((3,)))
        # original is unchanged
        np.testing.assert_allclose(outer.bias, jnp.zeros((3,)))

    def test_aset_nested_field(self):
        class Inner(drinx.DataClass):
            w: jax.Array

        @drinx.dataclass
        class Outer(drinx.DataClass):
            inner: Inner
            bias: jax.Array

        outer = Outer(inner=Inner(w=jnp.ones((3,))), bias=jnp.zeros((3,)))

        # Update a nested field
        outer3 = outer.aset("inner->w", jnp.zeros((3,)))

        assert isinstance(outer3, Outer)
        np.testing.assert_allclose(outer3.inner.w, jnp.zeros((3,)))
        # original is unchanged
        np.testing.assert_allclose(outer.inner.w, jnp.ones((3,)))


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJaxTransformsExample:
    def test_jit_with_updated_copy(self):
        class State(drinx.DataClass):
            x: jax.Array
            step_size: float = drinx.static_field(default=0.1)

        @jax.jit
        def update(state):
            return state.updated_copy(x=state.x - state.step_size)

        state = State(x=jnp.array([1.0, 2.0, 3.0]))
        result = update(state)
        assert isinstance(result, State)
        np.testing.assert_allclose(result.x, jnp.array([0.9, 1.9, 2.9]), atol=1e-6)

    def test_grad_same_structure_as_input(self):
        class State(drinx.DataClass):
            x: jax.Array
            step_size: float = drinx.static_field(default=0.1)

        def loss(state):
            return jnp.sum(state.x**2)

        grads = jax.grad(loss)(State(x=jnp.array([1.0, 2.0, 3.0])))

        # gradients have the same structure as the input
        assert isinstance(grads, State)
        np.testing.assert_allclose(grads.x, jnp.array([2.0, 4.0, 6.0]))

    def test_vmap_batch_over_dynamic_fields(self):
        class State(drinx.DataClass):
            x: jax.Array
            step_size: float = drinx.static_field(default=0.1)

        @jax.vmap
        def scale(state):
            return state.x * 2

        batched = State(x=jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        result = scale(batched)  # shape (2, 2)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, jnp.array([[2.0, 4.0], [6.0, 8.0]]))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


class TestVisualizationExample:
    def setup_method(self):
        class Encoder(drinx.DataClass):
            w: jax.Array
            b: jax.Array

        class Model(drinx.DataClass):
            encoder: Encoder
            head: jax.Array

        self.model = Model(
            encoder=Encoder(w=jnp.ones((16, 32)), b=jnp.zeros((16,))),
            head=jnp.ones((4, 16)),
        )

    def test_tree_diagram_returns_string(self):
        result = drinx.tree_diagram(self.model)
        assert isinstance(result, str)

    def test_tree_diagram_contains_structure(self):
        result = drinx.tree_diagram(self.model)
        assert "Model" in result
        assert ".encoder" in result
        assert ".w" in result
        assert ".b" in result
        assert ".head" in result

    def test_tree_diagram_contains_leaf_summaries(self):
        result = drinx.tree_diagram(self.model)
        assert "f32[16,32]" in result
        assert "f32[16]" in result
        assert "f32[4,16]" in result

    def test_tree_summary_returns_string(self):
        result = drinx.tree_summary(self.model)
        assert isinstance(result, str)

    def test_tree_summary_contains_paths(self):
        result = drinx.tree_summary(self.model)
        assert ".encoder" in result
        assert ".w" in result
        assert ".b" in result
        assert ".head" in result

    def test_tree_summary_contains_totals_row(self):
        result = drinx.tree_summary(self.model)
        assert "Σ" in result
        # total element count: 16*32 + 16 + 4*16 = 512 + 16 + 64 = 592
        assert "592" in result
