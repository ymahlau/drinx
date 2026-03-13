"""Tests for drinx.visualize: visualize_leaf and tree_diagram."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import drinx
from drinx import DataClass, static_field
from drinx.visualize import (
    _format_key,
    _get_one_level,
    tree_diagram,
    tree_summary,
    visualize_leaf,
)


# ---------------------------------------------------------------------------
# visualize_leaf — Python scalars
# ---------------------------------------------------------------------------


class TestVisualizeLeafPythonScalars:
    def test_bool_true(self):
        assert visualize_leaf(True) == "True"

    def test_bool_false(self):
        assert visualize_leaf(False) == "False"

    def test_int(self):
        assert visualize_leaf(42) == "42"

    def test_negative_int(self):
        assert visualize_leaf(-7) == "-7"

    def test_float(self):
        assert visualize_leaf(3.14) == "3.14"

    def test_complex(self):
        assert visualize_leaf(1 + 2j) == "(1+2j)"

    def test_zero_float(self):
        assert visualize_leaf(0.0) == "0.0"


# ---------------------------------------------------------------------------
# visualize_leaf — scalar arrays (0-d)
# ---------------------------------------------------------------------------


class TestVisualizeLeafScalarArrays:
    def test_float32_scalar(self):
        result = visualize_leaf(jnp.array(1.5, dtype=jnp.float32))
        assert result == "f32[] 1.5"

    def test_int32_scalar(self):
        result = visualize_leaf(jnp.array(7, dtype=jnp.int32))
        assert result == "i32[] 7"

    def test_bool_scalar_true(self):
        result = visualize_leaf(jnp.array(True))
        assert result == "bool[] True"

    def test_bool_scalar_false(self):
        result = visualize_leaf(jnp.array(False))
        assert result == "bool[] False"

    def test_numpy_scalar(self):
        # np.float64 is a Python scalar subclass, so visualize_leaf returns repr()
        result = visualize_leaf(np.float64(2.5))
        assert "2.5" in result


# ---------------------------------------------------------------------------
# visualize_leaf — empty arrays
# ---------------------------------------------------------------------------


class TestVisualizeLeafEmptyArrays:
    def test_1d_empty(self):
        result = visualize_leaf(jnp.array([], dtype=jnp.float32))
        assert result == "f32[0] (empty)"

    def test_2d_empty(self):
        result = visualize_leaf(jnp.zeros((0, 3), dtype=jnp.int32))
        assert result == "i32[0,3] (empty)"


# ---------------------------------------------------------------------------
# visualize_leaf — boolean arrays
# ---------------------------------------------------------------------------


class TestVisualizeLeafBoolArrays:
    def test_all_true(self):
        result = visualize_leaf(jnp.array([True, True, True]))
        assert result == "bool[3] #T=3, #F=0"

    def test_all_false(self):
        result = visualize_leaf(jnp.array([False, False]))
        assert result == "bool[2] #T=0, #F=2"

    def test_mixed(self):
        result = visualize_leaf(jnp.array([True, False, True, False, True]))
        assert result == "bool[5] #T=3, #F=2"

    def test_2d_bool(self):
        result = visualize_leaf(jnp.ones((2, 3), dtype=bool))
        assert result == "bool[2,3] #T=6, #F=0"


# ---------------------------------------------------------------------------
# visualize_leaf — numeric arrays
# ---------------------------------------------------------------------------


class TestVisualizeLeafNumericArrays:
    def test_float32_1d(self):
        result = visualize_leaf(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
        assert "f32[3]" in result
        assert "∈" in result
        assert "μ=" in result
        assert "σ=" in result

    def test_dtype_prefix_float64(self):
        result = visualize_leaf(np.array([1.0, 2.0], dtype=np.float64))
        assert result.startswith("f64[2]")

    def test_dtype_prefix_int8(self):
        result = visualize_leaf(np.array([1, 2, 3], dtype=np.int8))
        assert result.startswith("i8[3]")

    def test_dtype_prefix_uint16(self):
        result = visualize_leaf(np.array([0, 65535], dtype=np.uint16))
        assert result.startswith("u16[2]")

    def test_stats_correct(self):
        arr = jnp.array([1.0, 3.0], dtype=jnp.float32)
        result = visualize_leaf(arr)
        # min=1, max=3, mean=2, std=1
        assert "∈ [1, 3]" in result
        assert "μ=2" in result
        assert "σ=1" in result

    def test_2d_shape_in_prefix(self):
        result = visualize_leaf(jnp.ones((2, 4), dtype=jnp.float32))
        assert result.startswith("f32[2,4]")


# ---------------------------------------------------------------------------
# visualize_leaf — complex arrays
# ---------------------------------------------------------------------------


class TestVisualizeLeafComplexArrays:
    def test_complex_prefix_and_magnitude_symbol(self):
        arr = np.array([1 + 0j, 0 + 1j], dtype=np.complex64)
        result = visualize_leaf(arr)
        assert result.startswith("c64[2]")
        assert "|·|" in result
        assert "∈" in result

    def test_complex128(self):
        arr = np.array([3 + 4j], dtype=np.complex128)
        result = visualize_leaf(arr)
        assert result.startswith("c128[1]")
        # magnitude of 3+4j is 5
        assert "∈ [5, 5]" in result


# ---------------------------------------------------------------------------
# visualize_leaf — tracers
# ---------------------------------------------------------------------------


class TestVisualizeLeafTracers:
    def test_tracer_inside_jit(self):
        results = []

        @jax.jit
        def f(x):
            results.append(visualize_leaf(x))
            return x

        f(jnp.array([1.0, 2.0]))
        assert len(results) == 1
        assert "(Tracer)" in results[0]
        assert "f32[2]" in results[0]

    def test_tracer_inside_vmap(self):
        results = []

        @jax.vmap
        def f(x):
            results.append(visualize_leaf(x))
            return x

        f(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        assert len(results) == 1
        assert "(Tracer)" in results[0]


# ---------------------------------------------------------------------------
# visualize_leaf — unsupported type
# ---------------------------------------------------------------------------


class TestVisualizeLeafUnsupported:
    def test_string_returns_repr(self):
        assert visualize_leaf("hello") == "'hello'"  # type: ignore[arg-type]

    def test_list_returns_repr(self):
        assert visualize_leaf([1, 2, 3]) == "[1, 2, 3]"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _format_key
# ---------------------------------------------------------------------------


class TestFormatKey:
    def test_get_attr_key(self):
        key = jax.tree_util.GetAttrKey("foo")
        assert _format_key(key) == ".foo"

    def test_sequence_key(self):
        key = jax.tree_util.SequenceKey(3)
        assert _format_key(key) == "[3]"

    def test_dict_key_str(self):
        key = jax.tree_util.DictKey("mykey")
        assert _format_key(key) == "['mykey']"

    def test_dict_key_int(self):
        key = jax.tree_util.DictKey(42)
        assert _format_key(key) == "[42]"

    def test_flattened_index_key(self):
        key = jax.tree_util.FlattenedIndexKey(5)
        assert _format_key(key) == "[5]"

    def test_fallback_unknown_key(self):
        # Any object without a recognised type falls back to str()
        class _FakeKey:
            def __str__(self):
                return "FAKE"

        assert _format_key(_FakeKey()) == "FAKE"


# ---------------------------------------------------------------------------
# _get_one_level
# ---------------------------------------------------------------------------


class TestGetOneLevel:
    def test_plain_float_is_leaf(self):
        assert _get_one_level(1.0) is None

    def test_jax_array_is_leaf(self):
        assert _get_one_level(jnp.array([1.0, 2.0])) is None

    def test_list_children(self):
        result = _get_one_level([10.0, 20.0])
        assert result is not None
        assert len(result) == 2
        labels = [label for label, _ in result]
        assert labels == ["[0]", "[1]"]

    def test_tuple_children(self):
        result = _get_one_level((1.0, 2.0, 3.0))
        assert result is not None
        assert len(result) == 3

    def test_dict_children(self):
        result = _get_one_level({"a": 1.0, "b": 2.0})
        assert result is not None
        labels = [label for label, _ in result]
        assert "['a']" in labels
        assert "['b']" in labels

    def test_drinx_dataclass_children(self):
        class Pt(DataClass):
            x: float
            y: float

        result = _get_one_level(Pt(x=1.0, y=2.0))
        assert result is not None
        labels = [label for label, _ in result]
        assert labels == [".x", ".y"]

    def test_empty_list_returns_empty(self):
        result = _get_one_level([])
        # An empty list flattens to zero leaves — should be treated as non-leaf with no children
        assert result == [] or result is None  # either is acceptable


# ---------------------------------------------------------------------------
# tree_diagram — structure
# ---------------------------------------------------------------------------


class TestTreeDiagramStructure:
    def test_starts_with_class_name(self):
        result = tree_diagram(1.0)
        assert result.startswith("float")

    def test_single_leaf(self):
        result = tree_diagram(42.0)
        assert result == "float"

    def test_flat_list(self):
        result = tree_diagram([1.0, 2.0, 3.0])
        lines = result.splitlines()
        assert lines[0] == "list"
        assert len(lines) == 4  # class name + 3 children

    def test_last_child_uses_corner(self):
        result = tree_diagram([1.0, 2.0])
        lines = result.splitlines()
        assert lines[-1].startswith("└── ")

    def test_non_last_child_uses_tee(self):
        result = tree_diagram([1.0, 2.0])
        lines = result.splitlines()
        assert lines[1].startswith("├── ")

    def test_nested_prefix_continuation(self):
        result = tree_diagram([[1.0, 2.0], 3.0])
        lines = result.splitlines()
        # The nested children of [0] must use "│   " prefix
        assert any("│   " in line for line in lines)

    def test_last_branch_uses_spaces_not_pipe(self):
        result = tree_diagram([3.0, [1.0, 2.0]])
        lines = result.splitlines()
        # Children of last element must use "    " (4 spaces) prefix, not "│   "
        nested_lines = [line for line in lines if line.startswith("    ")]
        assert len(nested_lines) > 0


# ---------------------------------------------------------------------------
# tree_diagram — leaf formatting
# ---------------------------------------------------------------------------


class TestTreeDiagramLeafFormatting:
    def test_float_leaf(self):
        result = tree_diagram([1.5])
        assert "[0]=1.5" in result

    def test_array_leaf(self):
        result = tree_diagram([jnp.array([1.0, 2.0])])
        assert "f32[2]" in result

    def test_dict_leaf_labels(self):
        result = tree_diagram({"x": 1.0, "y": 2.0})
        assert "['x']=1.0" in result
        assert "['y']=2.0" in result

    def test_drinx_attr_labels(self):
        class Foo(DataClass):
            a: float
            b: float

        result = tree_diagram(Foo(a=1.0, b=2.0))
        assert ".a=1.0" in result
        assert ".b=2.0" in result


# ---------------------------------------------------------------------------
# tree_diagram — max_depth
# ---------------------------------------------------------------------------


class TestTreeDiagramMaxDepth:
    def test_max_depth_none_expands_fully(self):
        result = tree_diagram([[1.0, 2.0], [3.0, 4.0]])
        # All four leaves must appear
        assert result.count("=") == 4

    def test_max_depth_1_collapses_nested(self):
        result = tree_diagram([[1.0, 2.0], 3.0], max_depth=1)
        lines = result.splitlines()
        assert any("list ..." in line for line in lines)

    def test_max_depth_1_leaves_unchanged(self):
        result = tree_diagram([1.0, 2.0], max_depth=1)
        lines = result.splitlines()
        # Leaves at depth 0 are still shown normally
        assert any("[0]=1.0" in line for line in lines)
        assert any("[1]=2.0" in line for line in lines)

    def test_max_depth_2_expands_one_level(self):
        result = tree_diagram([[[1.0]], 2.0], max_depth=2)
        lines = result.splitlines()
        # depth-1 list is shown but not its children
        assert any("list ..." in line for line in lines)

    def test_max_depth_exact_boundary(self):
        # With max_depth=2 a two-level deep leaf should still appear
        result = tree_diagram([[1.0]], max_depth=2)
        assert "[0]=1.0" in result

    def test_max_depth_0_collapses_non_leaf_children(self):
        # max_depth=0: non-leaf children at depth 0 are collapsed; leaves still render normally
        result = tree_diagram([[1.0, 2.0], 3.0], max_depth=0)
        lines = result.splitlines()
        assert any("list ..." in line for line in lines)
        assert any("=3.0" in line for line in lines)

    def test_depth_label_shows_type_name(self):
        result = tree_diagram({"nested": [1.0, 2.0]}, max_depth=1)
        assert "list ..." in result


# ---------------------------------------------------------------------------
# tree_diagram — drinx dataclasses
# ---------------------------------------------------------------------------


class TestTreeDiagramDrinxDataclass:
    def test_simple_dataclass(self):
        class Pt(DataClass):
            x: float
            y: float

        result = tree_diagram(Pt(x=3.0, y=4.0))
        assert ".x=3.0" in result
        assert ".y=4.0" in result

    def test_nested_dataclass(self):
        class Inner(DataClass):
            v: float

        class Outer(DataClass):
            a: Inner
            b: float

        result = tree_diagram(Outer(a=Inner(v=1.0), b=2.0))
        assert ".a:Inner" in result
        assert ".v=1.0" in result
        assert ".b=2.0" in result

    def test_static_fields_excluded(self):
        class Cfg(DataClass):
            weights: jnp.ndarray
            lr: float = static_field(default=0.01)

        result = tree_diagram(Cfg(weights=jnp.array([1.0, 2.0])))
        # static fields are not pytree leaves — they won't appear in the diagram
        assert ".lr" not in result
        assert ".weights" in result

    def test_mixed_pytree_and_dataclass(self):
        class Foo(DataClass):
            a: float
            b: tuple

        result = tree_diagram(Foo(a=1.0, b=(10.0, 20.0)))
        assert ".a=1.0" in result
        assert ".b:tuple" in result
        assert "[0]=10.0" in result
        assert "[1]=20.0" in result

    def test_max_depth_with_dataclass(self):
        class Inner(DataClass):
            v: float

        class Outer(DataClass):
            a: Inner
            b: float

        result = tree_diagram(Outer(a=Inner(v=1.0), b=2.0), max_depth=1)
        assert ".a:Inner ..." in result
        assert ".b=2.0" in result
        assert ".v" not in result

    def test_decorator_style_dataclass(self):
        @drinx.dataclass
        class Params:
            w: jnp.ndarray
            b: float

        result = tree_diagram(Params(w=jnp.array([1.0]), b=0.5))
        assert ".w" in result
        assert ".b=0.5" in result


# ---------------------------------------------------------------------------
# tree_diagram — edge cases
# ---------------------------------------------------------------------------


class TestTreeDiagramEdgeCases:
    def test_empty_list(self):
        result = tree_diagram([])
        assert result.startswith("list")

    def test_deeply_nested(self):
        # Build a 5-level deep list: [[[[[1.0]]]]]
        tree = 1.0
        for _ in range(5):
            tree = [tree]
        result = tree_diagram(tree)
        lines = result.splitlines()
        # class name + 4 container lines + 1 leaf = 6 lines
        assert len(lines) == 6

    def test_wide_tree(self):
        result = tree_diagram(list(range(10)))
        lines = result.splitlines()
        assert len(lines) == 11  # class name + 10 leaves

    def test_returns_string(self):
        assert isinstance(tree_diagram([1.0, 2.0]), str)

    def test_single_element_list(self):
        result = tree_diagram([42.0])
        lines = result.splitlines()
        assert lines[0] == "list"
        assert "└── [0]=42.0" in lines[1]

    def test_dict_nested_in_list(self):
        result = tree_diagram([{"a": 1.0}])
        assert "['a']=1.0" in result


# ---------------------------------------------------------------------------
# tree_diagram — static_leaves
# ---------------------------------------------------------------------------


class TestTreeDiagramStaticLeaves:
    def test_static_field_shown_when_enabled(self):
        class Cfg(DataClass):
            weights: jnp.ndarray
            lr: float = static_field(default=0.01)

        result = tree_diagram(Cfg(weights=jnp.array([1.0, 2.0])), static_leaves=True)
        assert ".lr" in result

    def test_static_field_absent_by_default(self):
        class Cfg(DataClass):
            weights: jnp.ndarray
            lr: float = static_field(default=0.01)

        result = tree_diagram(Cfg(weights=jnp.array([1.0, 2.0])))
        assert ".lr" not in result

    def test_declaration_order_preserved(self):
        class Cfg(DataClass):
            a: float
            c: float
            b: float = static_field(default=0.5)

        result = tree_diagram(Cfg(a=1.0, c=3.0), static_leaves=True)
        lines = result.splitlines()
        labels = [line.split("=")[0].strip().lstrip("├└─ ") for line in lines[1:]]
        assert labels == [".a", ".c", ".b"]

    def test_static_int_value_rendered(self):
        class Cfg(DataClass):
            x: float
            n: int = static_field(default=4)

        result = tree_diagram(Cfg(x=1.0), static_leaves=True)
        assert ".n=4" in result

    def test_static_tuple_value_rendered(self):
        class Cfg(DataClass):
            x: float
            shape: tuple = static_field(default=(2, 3))

        result = tree_diagram(Cfg(x=1.0), static_leaves=True)
        # tuple is a pytree so it gets expanded
        assert ".shape:tuple" in result
        assert "[0]=2" in result
        assert "[1]=3" in result


# ---------------------------------------------------------------------------
# tree_summary
# ---------------------------------------------------------------------------


class TestTreeSummary:
    def test_returns_string(self):
        assert isinstance(tree_summary([1.0, 2.0]), str)

    def test_header_row_present(self):
        result = tree_summary([1.0])
        assert "Name" in result
        assert "Type" in result
        assert "Count" in result
        assert "Size" in result

    def test_summary_row_present(self):
        result = tree_summary([1.0])
        assert "Σ" in result
        assert "Tree" in result

    def test_python_scalar_no_size(self):
        result = tree_summary([1.0, 2.0])
        lines = result.splitlines()
        # Find the data rows (skip header and dividers)
        data_rows = [
            line
            for line in lines
            if line.startswith("│") and "Name" not in line and "Σ" not in line
        ]
        # Each scalar leaf should have no size entry (empty size column)
        for row in data_rows:
            cells = [c.strip() for c in row.strip("│").split("│")]
            assert cells[3] == ""  # Size column empty for scalars

    def test_array_leaf_has_size(self):
        arr = jnp.ones((3,), dtype=jnp.float32)
        result = tree_summary([arr])
        assert "12.00B" in result  # 3 * 4 bytes

    def test_array_leaf_count(self):
        arr = jnp.ones((4,), dtype=jnp.float32)
        result = tree_summary([arr])
        # Count column should show 4
        assert "│4" in result or "│4 " in result

    def test_total_count_and_size(self):
        arr = jnp.ones((3,), dtype=jnp.float32)
        result = tree_summary({"x": 1.0, "arr": arr})
        # total count = 1 (scalar) + 3 (array) = 4
        lines = result.splitlines()
        sigma_row = next(line for line in lines if "Σ" in line)
        cells = [c.strip() for c in sigma_row.strip("│").split("│")]
        assert cells[2] == "4"  # total count
        assert "12.00B" in cells[3]  # total size

    def test_dataclass_leaves(self):
        class Pt(DataClass):
            x: float
            y: jnp.ndarray

        pt = Pt(x=1.0, y=jnp.array([1.0, 2.0], dtype=jnp.float32))
        result = tree_summary(pt)
        assert ".x" in result
        assert ".y" in result
        assert "float" in result
        assert "f32[2]" in result

    def test_column_widths_adapt(self):
        # A very long path name should widen the Name column
        result = tree_summary({"a_very_long_key_name": 1.0})
        assert "a_very_long_key_name" in result

    def test_max_depth_none_expands_fully(self):
        result_default = tree_summary([[1.0, 2.0], 3.0])
        result_none = tree_summary([[1.0, 2.0], 3.0], max_depth=None)
        assert result_default == result_none
        # All three leaf paths present
        assert "[0][0]" in result_default
        assert "[0][1]" in result_default
        assert "[1]" in result_default

    def test_max_depth_1_aggregates_subtree(self):
        arr = jnp.ones((2,), dtype=jnp.float32)
        # Depth-1: [arr, arr] — each element is a leaf at depth 1
        result = tree_summary([[arr, arr], arr], max_depth=1)
        lines = result.splitlines()
        # The nested list should be a single aggregated row
        data_rows = [
            line
            for line in lines
            if line.startswith("│") and "Name" not in line and "Σ" not in line
        ]
        assert len(data_rows) == 2  # [0] aggregated + [1] leaf

    def test_max_depth_1_aggregated_count(self):
        arr = jnp.ones((3,), dtype=jnp.float32)
        result = tree_summary([[arr, arr], arr], max_depth=1)
        lines = result.splitlines()
        sigma_row = next(line for line in lines if "Σ" in line)
        cells = [c.strip() for c in sigma_row.strip("│").split("│")]
        # Total: 3+3+3 = 9 elements
        assert cells[2] == "9"

    def test_format_bytes_units(self):
        # 2KB array
        arr = jnp.ones((512,), dtype=jnp.float32)  # 512 * 4 = 2048 bytes
        result = tree_summary([arr])
        assert "KB" in result

    def test_pure_list_of_arrays(self):
        arrays = [jnp.ones((2,), dtype=jnp.float32) for _ in range(3)]
        result = tree_summary(arrays)
        lines = result.splitlines()
        data_rows = [
            line
            for line in lines
            if line.startswith("│") and "Name" not in line and "Σ" not in line
        ]
        assert len(data_rows) == 3  # one row per array leaf
