import dataclasses
from drinx.transform import dataclass
from typing import dataclass_transform, Any, Self
from dataclasses import field as orig_field
from drinx.attribute import field, static_field, private_field, static_private_field


@dataclass_transform(
    field_specifiers=(
        orig_field,
        field,
        static_field,
        private_field,
        static_private_field,
    )
)
class DataClass:
    """Base class alternative to the ``@drinx.dataclass`` decorator.

    Subclassing ``DataClass`` automatically applies the ``@drinx.dataclass``
    transform, registering the subclass as a frozen dataclass and a JAX pytree
    node.  Fields annotated with :func:`static_field` (or ``field(static=True)``)
    are placed in the pytree auxiliary data (not traced by JAX); all other fields
    become pytree leaves.

    Usage::

        class MyModel(DataClass):
            weights: jax.Array
            learning_rate: float = static_field(default=1e-3)

    Dataclass keyword arguments (``init``, ``repr``, ``eq``, etc.) can be
    forwarded via the class definition::

        class MyModel(DataClass, order=True, kw_only=True):
            ...

    Also provides :meth:`aset` for functional nested updates and
    :meth:`updated_copy` as a convenience wrapper around
    :func:`dataclasses.replace`.
    """

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
        """Apply the ``@drinx.dataclass`` transform to every subclass automatically.

        Called by Python whenever a new subclass of :class:`DataClass` is
        defined.  Accepts the same keyword arguments as the standard
        :func:`dataclasses.dataclass` decorator (except ``frozen``, which is
        always ``True``).

        Args:
            init: Generate ``__init__``.
            repr: Generate ``__repr__``.
            eq: Generate ``__eq__`` and ``__hash__``.
            order: Generate comparison methods (``<``, ``<=``, ``>``, ``>=``).
            unsafe_hash: Force generation of ``__hash__`` even when ``eq=True``.
            match_args: Set ``__match_args__`` for structural pattern matching.
            kw_only: Make all fields keyword-only in ``__init__``.
            slots: Generate ``__slots__`` (ignored; kept for API compatibility).
            weakref_slot: Add a ``__weakref__`` slot (ignored; kept for API
                compatibility).
        """
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

    @staticmethod
    def _parse_operations(s: str) -> list[tuple[str | int, str]]:
        """Parse a path string into a sequence of typed operations.

        Splits an ``aset``-style path such as ``"a->b->[0]->['key']"`` into an
        ordered list of ``(operand, operation_type)`` pairs understood by
        :meth:`aset`.

        Operation types:

        * ``"attribute"`` — attribute access (``getattr``).  Operand is the
          attribute name as a :class:`str`.
        * ``"index"`` — integer subscript (``obj[n]``).  Operand is an
          :class:`int`.
        * ``"key"`` — string subscript (``obj['k']``).  Operand is a
          :class:`str`.

        Args:
            s: Path string.  Steps are separated by ``"->"``.  Integer indices
               are written as ``[n]`` and string keys as ``['k']``.

        Returns:
            Ordered list of ``(operand, operation_type)`` pairs.

        Raises:
            ValueError: If *s* is empty, malformed, or contains invalid
                identifiers or bracket expressions.
        """
        if not s:
            raise ValueError("Empty string is not valid")

        operations = []
        i = 0

        while i < len(s):
            if i > 0:
                # Expect "->" separator
                if not s[i:].startswith("->"):
                    raise ValueError(f"Expected '->' at position {i}")
                i += 2  # Skip "->"

                if i >= len(s):
                    raise ValueError("String ends with '->'")

            # Parse the next operation
            if s[i] == "[":
                # Find the closing bracket
                j = i + 1
                while j < len(s) and s[j] != "]":
                    j += 1

                if j >= len(s):
                    raise ValueError(f"Unclosed bracket starting at position {i}")

                bracket_content = s[i + 1 : j].strip()

                # Determine if it's an integer or string
                if bracket_content.isdigit() or (
                    bracket_content.startswith("-") and bracket_content[1:].isdigit()
                ):
                    operations.append((int(bracket_content), "index"))
                elif bracket_content.startswith("'") and bracket_content.endswith("'"):
                    # Extract string content
                    if len(bracket_content) < 2:
                        raise ValueError(
                            f"Invalid string format in brackets: [{bracket_content}]"
                        )

                    string_content = bracket_content[1:-1]

                    # Check for forbidden characters
                    if "'" in string_content:
                        raise ValueError(
                            f"String keys cannot contain single quotes: '{string_content}'"
                        )
                    if "[" in string_content or "]" in string_content:
                        raise ValueError(
                            f"String keys cannot contain square brackets: '{string_content}'"
                        )

                    operations.append((string_content, "key"))
                else:
                    raise ValueError(f"Invalid bracket content: [{bracket_content}]")

                i = j + 1
            else:
                # Parse attribute name
                j = i
                while j < len(s) and s[j : j + 2] != "->":
                    j += 1

                attr_name = s[i:j]

                # Validate attribute name
                if not attr_name:
                    raise ValueError(f"Empty attribute at position {i}")

                # Check if it's a valid Python identifier
                if not attr_name.isidentifier():
                    raise ValueError(f"Invalid attribute name: '{attr_name}'")

                operations.append((attr_name, "attribute"))
                i = j

        return operations

    def aset(
        self,
        attr_name: str,
        val: Any,
        create_new_ok: bool = False,
    ) -> Self:
        """Sets an attribute of this class. In contrast to the classical .at[].set(), this method updates the class
        attribute directly and does not only operate on jax pytree leaf nodes. Instead, replaces the full attribute
        with the new value.

        The attribute can either be the attribute name of this class, or for nested classes it can also be the
        attribute name of a class, which itself is an attribute of this class. The syntax for this operation could
        look like this: "a->b->[0]->['name']". Here, the current class has an attribute a, which has an attribute b,
        which is a list, which we index at index 0, which is an element of type dictionary, which we index using
        the dictionary key 'name'.

        Note that dictionary keys cannot contain square brackets or single quotes (even if they are escaped).

        Args:
            attr_name (str): Name of attribute to set
            val (Any): Value to set the attribute to
            create_new_ok (bool, optional): If false (default), throw an error if the attribute does not exist.
                If true, creates a new attribute if the attribute name does not exist yet.

        Returns:
            Self: Updated instance with new attribute value
        """
        ops = self._parse_operations(attr_name)

        # 1. Top-down traversal: Find final attribute and save intermediate parents
        attr_list = [self]
        current_parent = self
        for idx, (op, op_type) in enumerate(ops):
            if op_type == "attribute":
                if not hasattr(current_parent, str(op)):
                    if idx != len(ops) - 1 or not create_new_ok:
                        raise Exception(
                            f"Attribute: {op} does not exist for {current_parent.__class__}"
                        )
                    current_parent = None
                else:
                    current_parent = getattr(current_parent, str(op))
            elif op_type == "index":
                if not hasattr(current_parent, "__getitem__"):
                    raise Exception(
                        f"{current_parent.__class__} does not implement __getitem__"
                    )
                current_parent = current_parent[int(op)]  # type: ignore
            elif op_type == "key":
                if not hasattr(current_parent, "__getitem__"):
                    raise Exception(
                        f"{current_parent.__class__} does not implement __getitem__"
                    )
                if op not in current_parent:  # type: ignore
                    if idx != len(ops) - 1 or not create_new_ok:
                        raise Exception(
                            f"Key: {op} does not exist for {current_parent}"
                        )
                    current_parent = None
                else:
                    current_parent = current_parent[op]  # type: ignore
            else:
                raise Exception(
                    f"Invalid operation type: {op_type}. This is an internal bug!"
                )

            if idx != len(ops) - 1:
                attr_list.append(current_parent)  # type: ignore

        # 2. Bottom-up copy: Set attributes functionally returning a brand-new top-level instance
        cur_attr = val
        for idx in list(range(len(attr_list)))[::-1]:
            op, op_type = ops[idx]
            current_parent = attr_list[idx]

            if op_type == "attribute":
                # Replaced generic DataClass check with standard dataclasses check
                if not dataclasses.is_dataclass(current_parent):
                    raise Exception(
                        f"Can only set attribute functionally on a dataclass, but got {current_parent.__class__}"
                    )

                # Use standard dataclasses.replace to functionally copy and update the frozen dataclass
                cur_attr = dataclasses.replace(current_parent, **{str(op): cur_attr})  # type: ignore

            elif op_type in ("index", "key"):
                if not hasattr(current_parent, "copy"):
                    raise Exception(
                        f"Target {current_parent.__class__} must implement a .copy() method for functional updates."
                    )

                # Copy the dictionary/list to avoid mutating the original frozen structure
                cpy = current_parent.copy()  # type: ignore

                if op_type == "index":
                    cpy[int(op)] = cur_attr
                else:
                    cpy[op] = cur_attr

                cur_attr = cpy
            else:
                raise Exception(
                    f"Invalid operation type: {op_type}. This is an internal bug!"
                )

        assert cur_attr.__class__ == self.__class__
        return cur_attr

    def updated_copy(self, **kwargs: Any) -> Self:
        """Returns an updated copy of the tree with modified top-level attributes.

        Args:
            **kwargs: Dictionary mapping immediate attribute names to their new values.

        Returns:
            Self: A newly instantiated object with the updated attributes.
        """
        # Directly utilize dataclasses.replace for standard functional updates
        return dataclasses.replace(self, **kwargs)  # ty:ignore[invalid-argument-type]
