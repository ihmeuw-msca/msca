from itertools import chain, filterfalse
from types import GenericAlias, UnionType
from typing import (
    Any,
    Generator,
    Hashable,
    Iterable,
    NoReturn,
    Self,
    Type,
    TypeVar,
    get_args,
)

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

T = TypeVar("T", bound=Hashable)


def unique_everseen(iterable: Iterable[T], seen: Iterable[T] = ()) -> Generator:
    seen = set(seen)
    for item in filterfalse(seen.__contains__, iterable):
        seen.add(item)
        yield item


def _unsupported_operand_type_error(operand: str, x: Any, y: Any) -> TypeError:
    return TypeError(
        "unsupported operand type(s) for {}: '{}' and '{}'".format(
            operand, type(x).__name__, type(y).__name__
        )
    )


def _wrong_number_of_arguments_error(
    cls: Type, wrong_item_type: Any
) -> TypeError:
    return TypeError(
        f"Wrong number of arguments for '{cls.__name__}', expected single item type, got '{wrong_item_type}'"
    )


def _unidentified_item_type_error(
    item_type: Any, expected_item_types: tuple[Type, ...]
) -> TypeError:
    type_names = ", ".join(
        [f"'{t.__module__}.{t.__name__}'" for t in expected_item_types]
    )
    return TypeError(
        f"Unrecognized item type '{item_type}', expected an instance of the following types: {type_names}"
    )


def _extract_types_from_item_type(
    item_type: Type | GenericAlias | UnionType, types: set[Type] | None = None
) -> list[Type]:
    types = set() if types is None else types
    if isinstance(item_type, Type):
        types.add(item_type)
        return types
    if isinstance(item_type, GenericAlias):
        types.add(item_type.__origin__)
    for item_sub_type in get_args(item_type):
        _extract_types_from_item_type(item_sub_type, types=types)
    return types


class UniqueSeq(tuple[T, ...]):
    def __new__(cls, iterable: Iterable[T] = ()) -> Self:
        if isinstance(iterable, cls):
            return iterable
        return super().__new__(cls, unique_everseen(iterable))

    def union(self, *iterables: Iterable[T]) -> Self:
        value = unique_everseen(chain(*iterables), seen=self)
        return super().__new__(type(self), chain(self, value))

    def __or__(self, value: Iterable[T]) -> Self:
        return self.union(value)

    def __add__(self, value: Any) -> NoReturn:
        raise _unsupported_operand_type_error("+", self, value)

    def __mul__(self, value: Any) -> NoReturn:
        raise _unsupported_operand_type_error("*", self, value)

    def __rmul__(self, value: Any) -> NoReturn:
        raise _unsupported_operand_type_error("*", value, self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({super().__repr__()})"

    def __class_getitem__(
        cls, item_type: Type | tuple[Type, ...]
    ) -> GenericAlias:
        if isinstance(item_type, tuple):
            if len(item_type) != 1:
                raise _wrong_number_of_arguments_error(cls, item_type)
            item_type = item_type[0]

        if item_type is not None:
            expected_item_types = (Type, GenericAlias, UnionType)
            if not isinstance(item_type, expected_item_types):
                raise _unidentified_item_type_error(
                    item_type, expected_item_types
                )
            bound = T.__bound__
            extracted_types = _extract_types_from_item_type(item_type)
            if not all(issubclass(t, bound) for t in extracted_types):
                raise TypeError(
                    f"Item type '{item_type}' is not a subclass of '{str(bound)}'"
                )
        return super().__class_getitem__(item_type)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        instance_schema = core_schema.is_instance_schema(cls)

        item_type = get_args(source) or T.__bound__
        iterable_t_schema = handler.generate_schema(Iterable[item_type])
        non_instance_schema = core_schema.no_info_after_validator_function(
            cls, iterable_t_schema
        )
        return core_schema.union_schema([instance_schema, non_instance_schema])
