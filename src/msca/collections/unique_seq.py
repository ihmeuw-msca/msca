from collections.abc import Hashable, Iterable, Iterator
from itertools import chain, filterfalse
from types import GenericAlias
from typing import Any, Never, Self, TypeVar, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

T = TypeVar("T", bound=Hashable)


def unique_everseen(
    iterable: Iterable[T], seen: Iterable[T] = ()
) -> Iterator[T]:
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

    def __add__(self, value: Any) -> Never:
        raise _unsupported_operand_type_error("+", self, value)

    def __mul__(self, value: Any) -> Never:
        raise _unsupported_operand_type_error("*", self, value)

    def __rmul__(self, value: Any) -> Never:
        raise _unsupported_operand_type_error("*", value, self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({super().__repr__()})"

    def __str__(self) -> str:
        return super().__repr__()

    def __class_getitem__(cls, item_type: Any) -> GenericAlias:
        if isinstance(item_type, tuple) and len(item_type) != 1:
            raise TypeError(
                f"Wrong number of parameters for '{cls.__name__}', "
                f"actual {len(item_type)}, expected 1"
            )
        return super().__class_getitem__(item_type)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        instance_schema = core_schema.is_instance_schema(cls)

        iterable_schema: dict[str, Any] = {"type": "generator"}
        if item_type := get_args(source):
            iterable_schema["items_schema"] = handler(item_type[0])
        non_instance_schema = core_schema.no_info_after_validator_function(
            cls, iterable_schema
        )
        return core_schema.union_schema([instance_schema, non_instance_schema])
