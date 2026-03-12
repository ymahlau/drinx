from drinx.transform import dataclass
from typing import dataclass_transform
from dataclasses import field as orig_field
from drinx.attribute import field, static_field, private_field, static_private_field

@dataclass_transform(field_specifiers=(orig_field, field, static_field, private_field, static_private_field))
class DataClass:
    """Base class providing advanced tree operations and automatic dataclass/JAX integration."""

    def __init_subclass__(
        cls,
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
    ):
        super().__init_subclass__()
        # Programmatically apply our custom dataclass wrapper to the subclass.
        dataclass_transform = dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            match_args=match_args,
            kw_only=kw_only,
            weakref_slot=weakref_slot,
        )
        dataclass_transform(cls)
        
    def tmp(self,):
        print("test")

    # def updated_copy(self, **kwargs: Any) -> Self:
    #     init_args = {}
    #     for f in self.get_class_fields(): # Assuming get_class_fields is defined
    #         if f.init:
    #             init_args[f.name] = getattr(self, f.name)
    #     init_args.update(kwargs)
    #     return self.__class__(**init_args)

    # def aset(self, attr_name: str, val: Any, create_new_ok: bool = False) -> Self:
    #     pass
        
    # def _aset(self, attr_name: str, val: Any):
    #     setattr(self, attr_name, val)
        
    # @staticmethod
    # def _parse_operations(s: str) -> list[tuple[str, str]]:
    #     pass