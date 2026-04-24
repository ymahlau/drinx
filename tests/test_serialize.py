"""Tests for drinx JSON serialization: to_dict, from_dict, to_json, from_json."""

from __future__ import annotations

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import drinx
from drinx import DataClass, dataclass, private_field, static_field
from drinx.serialize import (
    _deserialize,
    _serialize,
    from_dict,
    from_json,
    to_dict,
    to_json,
)

# ---------------------------------------------------------------------------
# Module-level fixture classes (required for round-trip tests — from_dict
# must importlib.import_module(cls.__module__) to reconstruct the class)
# ---------------------------------------------------------------------------


@dataclass
class _Pair:
    x: float
    y: float


@dataclass
class _WithStatic:
    lr: float
    size: int = static_field(default=10)


@dataclass
class _WithPrivate:
    x: float
    _cache: float = private_field(default=0.0)


class _WithComputed(DataClass):
    x: float
    _doubled: float = private_field(default=0.0)

    def __post_init__(self) -> None:
        self.aset_inplace("_doubled", self.x * 2)


@dataclass
class _Inner:
    val: float


@dataclass
class _Outer:
    inner: _Inner
    label: str = static_field(default="outer")


@dataclass
class _WithArray:
    weights: Any


@dataclass
class _WithDtype:
    dtype: Any = static_field(default=jnp.float32)


@dataclass
class _WithTuple:
    shape: tuple


@dataclass
class _WithDict:
    params: dict


@dataclass
class _WithList:
    items: list


@dataclass
class _WithBool:
    flag: bool


@dataclass
class _WithOptional:
    x: float | None = None


class _DataClassSubclass(DataClass):
    x: float
    n: int = static_field(default=5)


# ---------------------------------------------------------------------------
# TestSerializePrimitives
# ---------------------------------------------------------------------------


class TestSerializePrimitives:
    def test_none(self):
        assert _serialize(None) is None

    def test_true(self):
        result = _serialize(True)
        assert result is True
        assert type(result) is bool

    def test_false(self):
        result = _serialize(False)
        assert result is False
        assert type(result) is bool

    def test_bool_not_cast_to_int(self):
        assert type(_serialize(True)) is bool
        assert type(_serialize(False)) is bool

    def test_int(self):
        assert _serialize(42) == 42

    def test_negative_int(self):
        assert _serialize(-7) == -7

    def test_float(self):
        assert _serialize(3.14) == pytest.approx(3.14)

    def test_string(self):
        assert _serialize("hello world") == "hello world"

    def test_empty_string(self):
        assert _serialize("") == ""


# ---------------------------------------------------------------------------
# TestSerializeContainers
# ---------------------------------------------------------------------------


class TestSerializeContainers:
    def test_empty_list(self):
        assert _serialize([]) == []

    def test_list_of_primitives(self):
        assert _serialize([1, 2, 3]) == [1, 2, 3]

    def test_list_of_strings(self):
        assert _serialize(["a", "b"]) == ["a", "b"]

    def test_empty_tuple(self):
        assert _serialize(()) == {"__tuple__": []}

    def test_tuple_of_primitives(self):
        assert _serialize((1, 2, 3)) == {"__tuple__": [1, 2, 3]}

    def test_empty_dict(self):
        assert _serialize({}) == {"__dict__": {}}

    def test_dict_string_keys(self):
        assert _serialize({"a": 1, "b": 2}) == {"__dict__": {"a": 1, "b": 2}}

    def test_nested_list_in_list(self):
        assert _serialize([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_list_containing_tuple(self):
        assert _serialize([(1, 2)]) == [{"__tuple__": [1, 2]}]

    def test_tuple_containing_list(self):
        assert _serialize(([1, 2],)) == {"__tuple__": [[1, 2]]}

    def test_tuple_containing_dict(self):
        assert _serialize(({"a": 1},)) == {"__tuple__": [{"__dict__": {"a": 1}}]}

    def test_non_string_dict_key_raises(self):
        with pytest.raises(TypeError, match="string-keyed"):
            _serialize({1: "a"})

    def test_mixed_key_dict_raises(self):
        with pytest.raises(TypeError):
            _serialize({"a": 1, 2: "b"})


# ---------------------------------------------------------------------------
# TestSerializeArrays
# ---------------------------------------------------------------------------


class TestSerializeArrays:
    def test_1d_float32_structure(self):
        arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        result = _serialize(arr)
        assert result["__jax_array__"] is True
        assert result["__dtype__"] == "float32"
        assert result["__shape__"] == [3]
        assert result["__data__"] == pytest.approx([1.0, 2.0, 3.0])

    def test_2d_float32_shape(self):
        arr = jnp.ones((2, 3), dtype=jnp.float32)
        result = _serialize(arr)
        assert result["__shape__"] == [2, 3]
        assert len(result["__data__"]) == 2
        assert len(result["__data__"][0]) == 3

    def test_0d_array(self):
        arr = jnp.array(3.14, dtype=jnp.float32)
        result = _serialize(arr)
        assert result["__shape__"] == []
        assert result["__data__"] == pytest.approx(3.14)

    def test_int32_array(self):
        arr = jnp.array([0, 1, 2], dtype=jnp.int32)
        result = _serialize(arr)
        assert result["__dtype__"] == "int32"
        assert result["__data__"] == [0, 1, 2]

    def test_bool_array(self):
        arr = jnp.array([True, False, True])
        result = _serialize(arr)
        assert result["__dtype__"] == "bool"

    def test_bfloat16_array(self):
        arr = jnp.ones(3, dtype=jnp.bfloat16)
        result = _serialize(arr)
        assert result["__dtype__"] == "bfloat16"

    def test_complex64_array_has_real_imag(self):
        arr = jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64)
        result = _serialize(arr)
        assert isinstance(result["__data__"], dict)
        assert "__real__" in result["__data__"]
        assert "__imag__" in result["__data__"]

    def test_complex64_real_imag_values(self):
        arr = jnp.array([1 + 2j], dtype=jnp.complex64)
        result = _serialize(arr)
        assert result["__data__"]["__real__"] == pytest.approx([1.0])
        assert result["__data__"]["__imag__"] == pytest.approx([2.0])

    def test_numpy_ndarray(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = _serialize(arr)
        assert result["__jax_array__"] is True
        assert result["__dtype__"] == "float64"

    def test_data_is_json_serializable(self):
        arr = jnp.ones((2, 2), dtype=jnp.float32)
        result = _serialize(arr)
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# TestSerializeDtypeTypes
# ---------------------------------------------------------------------------


class TestSerializeDtypeTypes:
    def test_jnp_float32(self):
        assert _serialize(jnp.float32) == {"__jax_dtype__": "float32"}

    def test_jnp_float64(self):
        assert _serialize(jnp.float64) == {"__jax_dtype__": "float64"}

    def test_jnp_int32(self):
        assert _serialize(jnp.int32) == {"__jax_dtype__": "int32"}

    def test_jnp_bool_(self):
        # jnp.bool_.__name__ is "bool" in JAX >= 0.4 (_ScalarMeta convention)
        assert _serialize(jnp.bool_) == {"__jax_dtype__": "bool"}

    def test_np_float32(self):
        assert _serialize(np.float32) == {"__jax_dtype__": "float32"}

    def test_np_int64(self):
        assert _serialize(np.int64) == {"__jax_dtype__": "int64"}

    def test_bfloat16(self):
        result = _serialize(jnp.bfloat16)
        assert result == {"__jax_dtype__": "bfloat16"}


# ---------------------------------------------------------------------------
# TestSerializeDataclass
# ---------------------------------------------------------------------------


class TestSerializeDataclass:
    def test_has_module_name_value_keys(self):
        result = to_dict(_Pair(x=1.0, y=2.0))
        assert "__module__" in result
        assert "__name__" in result
        assert "__value__" in result

    def test_name_is_class_name(self):
        result = to_dict(_Pair(x=1.0, y=2.0))
        assert result["__name__"] == "_Pair"

    def test_init_fields_in_value(self):
        result = to_dict(_Pair(x=1.0, y=2.0))
        assert set(result["__value__"].keys()) == {"x", "y"}

    def test_static_init_field_included(self):
        result = to_dict(_WithStatic(lr=0.01, size=64))
        assert "lr" in result["__value__"]
        assert "size" in result["__value__"]

    def test_private_field_excluded(self):
        result = to_dict(_WithPrivate(x=1.0))
        assert "_cache" not in result["__value__"]
        assert "x" in result["__value__"]

    def test_nested_dataclass_value_is_dict(self):
        result = to_dict(_Outer(inner=_Inner(val=1.5), label="test"))
        inner_dict = result["__value__"]["inner"]
        assert isinstance(inner_dict, dict)
        assert inner_dict["__name__"] == "_Inner"
        assert inner_dict["__value__"]["val"] == pytest.approx(1.5)

    def test_local_class_raises_not_implemented(self):
        @dataclass
        class LocalFoo:
            x: int

        with pytest.raises(NotImplementedError):
            to_dict(LocalFoo(x=1))

    def test_class_object_not_instance_raises(self):
        with pytest.raises(TypeError):
            to_dict(_Pair)

    def test_non_serializable_field_raises(self):
        @dataclass
        class WithBadField:
            x: object

        class Unserializable:
            pass

        with pytest.raises((TypeError, NotImplementedError)):
            to_dict(WithBadField(x=Unserializable()))

    def test_plain_int_raises(self):
        with pytest.raises(TypeError):
            to_dict(42)

    def test_dataclass_subclass(self):
        result = to_dict(_DataClassSubclass(x=3.0, n=7))
        assert result["__name__"] == "_DataClassSubclass"
        assert set(result["__value__"].keys()) == {"x", "n"}


# ---------------------------------------------------------------------------
# TestDeserializePrimitives
# ---------------------------------------------------------------------------


class TestDeserializePrimitives:
    def test_none(self):
        assert _deserialize(None) is None

    def test_true(self):
        result = _deserialize(True)
        assert result is True
        assert type(result) is bool

    def test_false(self):
        result = _deserialize(False)
        assert result is False
        assert type(result) is bool

    def test_int(self):
        assert _deserialize(42) == 42

    def test_float(self):
        assert _deserialize(3.14) == pytest.approx(3.14)

    def test_string(self):
        assert _deserialize("hello") == "hello"


# ---------------------------------------------------------------------------
# TestDeserializeContainers
# ---------------------------------------------------------------------------


class TestDeserializeContainers:
    def test_list_passthrough(self):
        result = _deserialize([1, 2, 3])
        assert result == [1, 2, 3]
        assert type(result) is list

    def test_empty_list(self):
        result = _deserialize([])
        assert result == []
        assert type(result) is list

    def test_tuple_from_marker(self):
        result = _deserialize({"__tuple__": [1, 2, 3]})
        assert result == (1, 2, 3)
        assert type(result) is tuple

    def test_empty_tuple(self):
        result = _deserialize({"__tuple__": []})
        assert result == ()
        assert type(result) is tuple

    def test_dict_from_marker(self):
        result = _deserialize({"__dict__": {"a": 1, "b": 2}})
        assert result == {"a": 1, "b": 2}
        assert type(result) is dict

    def test_empty_dict(self):
        result = _deserialize({"__dict__": {}})
        assert result == {}
        assert type(result) is dict

    def test_nested_list_with_tuple(self):
        result = _deserialize([{"__tuple__": [1, 2]}, {"__tuple__": [3, 4]}])
        assert result == [(1, 2), (3, 4)]

    def test_unknown_dict_keys_raises(self):
        with pytest.raises(ValueError):
            _deserialize({"unknown_key": 123})

    def test_unknown_dict_multi_keys_raises(self):
        with pytest.raises(ValueError):
            _deserialize({"__tuple__": [], "extra": 1})


# ---------------------------------------------------------------------------
# TestDeserializeArrays
# ---------------------------------------------------------------------------


class TestDeserializeArrays:
    def test_1d_float32(self):
        data = {
            "__jax_array__": True,
            "__dtype__": "float32",
            "__shape__": [3],
            "__data__": [1.0, 2.0, 3.0],
        }
        result = _deserialize(data)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.float32
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_2d_int32(self):
        data = {
            "__jax_array__": True,
            "__dtype__": "int32",
            "__shape__": [2, 2],
            "__data__": [[1, 2], [3, 4]],
        }
        result = _deserialize(data)
        assert result.dtype == jnp.int32
        assert result.shape == (2, 2)

    def test_0d_array(self):
        data = {
            "__jax_array__": True,
            "__dtype__": "float32",
            "__shape__": [],
            "__data__": 3.14,
        }
        result = _deserialize(data)
        assert isinstance(result, jax.Array)
        assert result.shape == ()

    def test_bfloat16(self):
        data = {
            "__jax_array__": True,
            "__dtype__": "bfloat16",
            "__shape__": [2],
            "__data__": [1.0, 2.0],
        }
        result = _deserialize(data)
        assert result.dtype == jnp.bfloat16

    def test_complex64(self):
        data = {
            "__jax_array__": True,
            "__dtype__": "complex64",
            "__shape__": [2],
            "__data__": {"__real__": [1.0, 3.0], "__imag__": [2.0, 4.0]},
        }
        result = _deserialize(data)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.complex64
        assert jnp.allclose(result, jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64))

    def test_result_is_jax_array(self):
        data = {
            "__jax_array__": True,
            "__dtype__": "float32",
            "__shape__": [2],
            "__data__": [0.0, 1.0],
        }
        result = _deserialize(data)
        assert isinstance(result, jax.Array)


# ---------------------------------------------------------------------------
# TestDeserializeDtypeTypes
# ---------------------------------------------------------------------------


class TestDeserializeDtypeTypes:
    def test_float32(self):
        result = _deserialize({"__jax_dtype__": "float32"})
        assert result is jnp.float32

    def test_int32(self):
        result = _deserialize({"__jax_dtype__": "int32"})
        assert result is jnp.int32

    def test_bfloat16(self):
        result = _deserialize({"__jax_dtype__": "bfloat16"})
        assert result is jnp.bfloat16

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown JAX dtype"):
            _deserialize({"__jax_dtype__": "not_a_dtype"})


# ---------------------------------------------------------------------------
# TestDeserializeDataclass
# ---------------------------------------------------------------------------


class TestDeserializeDataclass:
    def test_simple_roundtrip(self):
        obj = _Pair(x=1.0, y=2.0)
        assert from_dict(to_dict(obj)) == obj

    def test_with_static_roundtrip(self):
        obj = _WithStatic(lr=0.001, size=64)
        assert from_dict(to_dict(obj)) == obj

    def test_private_field_reconstructed_by_default(self):
        obj = _WithPrivate(x=1.0)
        result = from_dict(to_dict(obj))
        assert result.x == pytest.approx(1.0)
        assert result._cache == pytest.approx(0.0)

    def test_computed_private_field_reconstructed(self):
        obj = _WithComputed(x=3.0)
        result = from_dict(to_dict(obj))
        assert result.x == pytest.approx(3.0)
        assert result._doubled == pytest.approx(6.0)

    def test_nested_dataclass_roundtrip(self):
        obj = _Outer(inner=_Inner(val=1.5), label="test")
        assert from_dict(to_dict(obj)) == obj

    def test_result_type_correct(self):
        obj = _Pair(x=1.0, y=2.0)
        result = from_dict(to_dict(obj))
        assert type(result) is _Pair

    def test_unknown_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            from_dict(
                {
                    "__module__": "nonexistent.module.xyz",
                    "__name__": "Foo",
                    "__value__": {},
                }
            )

    def test_unknown_class_in_valid_module_raises(self):
        with pytest.raises(AttributeError):
            from_dict(
                {"__module__": "drinx", "__name__": "NoSuchClass123", "__value__": {}}
            )

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError, match="not a dataclass"):
            from_dict({"__module__": "builtins", "__name__": "list", "__value__": {}})


# ---------------------------------------------------------------------------
# TestToJsonFromJson
# ---------------------------------------------------------------------------


class TestToJsonFromJson:
    def test_to_json_returns_str(self):
        assert isinstance(to_json(_Pair(x=1.0, y=2.0)), str)

    def test_to_json_is_valid_json(self):
        s = to_json(_Pair(x=1.0, y=2.0))
        json.loads(s)  # must not raise

    def test_pair_roundtrip(self):
        obj = _Pair(x=1.0, y=2.0)
        assert from_json(to_json(obj)) == obj

    def test_with_static_roundtrip(self):
        obj = _WithStatic(lr=0.001, size=32)
        assert from_json(to_json(obj)) == obj

    def test_nested_roundtrip(self):
        obj = _Outer(inner=_Inner(val=2.5), label="nested")
        assert from_json(to_json(obj)) == obj

    def test_with_array_roundtrip(self):
        obj = _WithArray(weights=jnp.ones(4, dtype=jnp.float32))
        result = from_json(to_json(obj))
        assert result.weights.dtype == jnp.float32
        assert jnp.allclose(result.weights, obj.weights)

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            from_json("{not valid json}")

    def test_to_json_with_tuple_field(self):
        obj = _WithTuple(shape=(2, 3, 4))
        result = from_json(to_json(obj))
        assert result.shape == (2, 3, 4)
        assert type(result.shape) is tuple

    def test_to_json_with_dict_field(self):
        obj = _WithDict(params={"lr": 0.01, "epochs": 10})
        result = from_json(to_json(obj))
        assert result.params == {"lr": 0.01, "epochs": 10}

    def test_to_json_with_list_field(self):
        obj = _WithList(items=[1.0, 2.0, 3.0])
        result = from_json(to_json(obj))
        assert result.items == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# TestPublicAPIExports
# ---------------------------------------------------------------------------


class TestPublicAPIExports:
    def test_drinx_has_to_dict(self):
        assert hasattr(drinx, "to_dict")

    def test_drinx_has_from_dict(self):
        assert hasattr(drinx, "from_dict")

    def test_drinx_has_to_json(self):
        assert hasattr(drinx, "to_json")

    def test_drinx_has_from_json(self):
        assert hasattr(drinx, "from_json")

    def test_to_dict_in_all(self):
        assert "to_dict" in drinx.__all__

    def test_from_dict_in_all(self):
        assert "from_dict" in drinx.__all__

    def test_to_json_in_all(self):
        assert "to_json" in drinx.__all__

    def test_from_json_in_all(self):
        assert "from_json" in drinx.__all__

    def test_importable_from_drinx_directly(self):
        from drinx import from_dict, from_json, to_dict, to_json  # noqa: F401


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_with_array_field(self):
        obj = _WithArray(weights=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
        result = from_json(to_json(obj))
        assert jnp.allclose(result.weights, obj.weights)
        assert result.weights.dtype == obj.weights.dtype
        assert result.weights.shape == obj.weights.shape

    def test_with_dtype_static_field(self):
        obj = _WithDtype(dtype=jnp.bfloat16)
        result = from_json(to_json(obj))
        assert result.dtype is jnp.bfloat16

    def test_dataclass_subclass_pattern(self):
        obj = _DataClassSubclass(x=2.5, n=3)
        result = from_json(to_json(obj))
        assert result == obj
        assert type(result) is _DataClassSubclass

    def test_deeply_nested(self):
        obj = _Outer(inner=_Inner(val=42.0), label="deep")
        result = from_json(to_json(obj))
        assert result == obj
        assert result.inner.val == pytest.approx(42.0)

    def test_tuple_field_preserves_type(self):
        obj = _WithTuple(shape=(3, 4, 5))
        result = from_json(to_json(obj))
        assert type(result.shape) is tuple
        assert result.shape == (3, 4, 5)

    def test_dict_field_roundtrip(self):
        obj = _WithDict(params={"hidden": 128, "dropout": 0.1})
        result = from_json(to_json(obj))
        assert result.params == obj.params

    def test_list_of_nested_dicts(self):
        obj = _WithList(items=[{"a": 1}, {"b": 2}])
        result = from_json(to_json(obj))
        assert result.items == [{"a": 1}, {"b": 2}]

    def test_2d_array_shape_and_values_preserved(self):
        arr = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        obj = _WithArray(weights=arr)
        result = from_json(to_json(obj))
        assert result.weights.shape == (2, 3)
        assert jnp.allclose(result.weights, arr)

    def test_complex64_array_roundtrip(self):
        arr = jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64)
        obj = _WithArray(weights=arr)
        result = from_json(to_json(obj))
        assert result.weights.dtype == jnp.complex64
        assert jnp.allclose(result.weights, arr)

    def test_private_field_excluded_then_reconstructed(self):
        obj = _WithPrivate(x=5.0)
        serialized = to_dict(obj)
        assert "_cache" not in serialized["__value__"]
        restored = from_dict(serialized)
        assert restored.x == pytest.approx(5.0)
        assert restored._cache == pytest.approx(0.0)

    def test_computed_private_field_excluded_then_reconstructed(self):
        obj = _WithComputed(x=4.0)
        serialized = to_dict(obj)
        assert "_doubled" not in serialized["__value__"]
        restored = from_dict(serialized)
        assert restored._doubled == pytest.approx(8.0)

    def test_bool_field_preserved(self):
        result = to_dict(_WithBool(flag=True))
        assert result["__value__"]["flag"] is True

    def test_none_field_roundtrip(self):
        result = to_dict(_WithOptional())
        assert result["__value__"]["x"] is None
