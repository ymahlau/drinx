"""Tests for drinx.field: field, static_field, private_field, static_private_field."""

import dataclasses

from drinx.field import field, private_field, static_field, static_private_field


class TestField:
    def test_default_is_dynamic(self):
        f = field()
        assert f.metadata["jax_static"] is False

    def test_static_false_is_dynamic(self):
        f = field(static=False)
        assert f.metadata["jax_static"] is False

    def test_static_true(self):
        f = field(static=True)
        assert f.metadata["jax_static"] is True

    def test_default_value(self):
        f = field(default=42)
        assert f.default == 42

    def test_default_factory(self):
        f = field(default_factory=list)
        assert f.default_factory is list

    def test_repr_false(self):
        f = field(repr=False)
        assert f.repr is False

    def test_repr_true_by_default(self):
        f = field()
        assert f.repr is True

    def test_compare_false(self):
        f = field(compare=False)
        assert f.compare is False

    def test_compare_true_by_default(self):
        f = field()
        assert f.compare is True

    def test_hash_true(self):
        f = field(hash=True)
        assert f.hash is True

    def test_hash_none_by_default(self):
        f = field()
        assert f.hash is None

    def test_init_false(self):
        f = field(init=False, default=0)
        assert f.init is False

    def test_init_true_by_default(self):
        f = field()
        assert f.init is True

    def test_preserves_existing_metadata(self):
        f = field(metadata={"custom_key": "custom_value"})
        assert f.metadata["custom_key"] == "custom_value"
        assert f.metadata["jax_static"] is False

    def test_jax_static_overrides_existing_metadata_key(self):
        # If someone passes jax_static in metadata, field() overwrites it with `static`
        f = field(metadata={"jax_static": True}, static=False)
        assert f.metadata["jax_static"] is False

    def test_metadata_none_is_treated_as_empty(self):
        f = field(metadata=None)
        assert "jax_static" in f.metadata

    def test_returns_dataclass_field(self):
        f = field()
        assert isinstance(f, dataclasses.Field)

    def test_default_and_static(self):
        f = field(default=99, static=True)
        assert f.default == 99
        assert f.metadata["jax_static"] is True


class TestStaticField:
    def test_is_static(self):
        f = static_field()
        assert f.metadata["jax_static"] is True

    def test_init_true_by_default(self):
        f = static_field()
        assert f.init is True

    def test_with_default(self):
        f = static_field(default=10)
        assert f.default == 10
        assert f.metadata["jax_static"] is True

    def test_with_default_factory(self):
        f = static_field(default_factory=dict)
        assert f.default_factory is dict
        assert f.metadata["jax_static"] is True

    def test_repr_false(self):
        f = static_field(repr=False)
        assert f.repr is False

    def test_compare_false(self):
        f = static_field(compare=False)
        assert f.compare is False

    def test_hash_true(self):
        f = static_field(hash=True)
        assert f.hash is True

    def test_preserves_metadata(self):
        f = static_field(metadata={"info": "test"})
        assert f.metadata["info"] == "test"
        assert f.metadata["jax_static"] is True

    def test_returns_dataclass_field(self):
        f = static_field()
        assert isinstance(f, dataclasses.Field)


class TestPrivateField:
    def test_init_is_false(self):
        f = private_field(default=0)
        assert f.init is False

    def test_dynamic_by_default(self):
        f = private_field(default=0)
        assert f.metadata["jax_static"] is False

    def test_can_be_static(self):
        f = private_field(default=0, static=True)
        assert f.metadata["jax_static"] is True
        assert f.init is False

    def test_with_default(self):
        f = private_field(default=7)
        assert f.default == 7
        assert f.init is False

    def test_with_default_factory(self):
        f = private_field(default_factory=set)
        assert f.default_factory is set
        assert f.init is False

    def test_repr_false(self):
        f = private_field(default=0, repr=False)
        assert f.repr is False

    def test_compare_false(self):
        f = private_field(default=0, compare=False)
        assert f.compare is False

    def test_preserves_metadata(self):
        f = private_field(default=0, metadata={"note": "private"})
        assert f.metadata["note"] == "private"
        assert f.metadata["jax_static"] is False

    def test_returns_dataclass_field(self):
        f = private_field(default=0)
        assert isinstance(f, dataclasses.Field)


class TestStaticPrivateField:
    def test_is_static(self):
        f = static_private_field(default=0)
        assert f.metadata["jax_static"] is True

    def test_init_is_false(self):
        f = static_private_field(default=0)
        assert f.init is False

    def test_with_default(self):
        f = static_private_field(default=42)
        assert f.default == 42
        assert f.metadata["jax_static"] is True
        assert f.init is False

    def test_with_default_factory(self):
        f = static_private_field(default_factory=tuple)
        assert f.default_factory is tuple
        assert f.metadata["jax_static"] is True
        assert f.init is False

    def test_repr_false(self):
        f = static_private_field(default=0, repr=False)
        assert f.repr is False

    def test_compare_false(self):
        f = static_private_field(default=0, compare=False)
        assert f.compare is False

    def test_preserves_metadata(self):
        f = static_private_field(default=0, metadata={"tag": "sp"})
        assert f.metadata["tag"] == "sp"
        assert f.metadata["jax_static"] is True

    def test_returns_dataclass_field(self):
        f = static_private_field(default=0)
        assert isinstance(f, dataclasses.Field)
