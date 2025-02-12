import pytest
from pydantic import BaseModel, ValidationError

from msca.collections import UniqueSeq


# All instances of Iterable[Hashable] should be accepted
@pytest.mark.parametrize(
    "x", [[1, 2, 3], (1, 2, 3), iter(["a", "b", "c", "c"])]
)
def test_new_success(x) -> None:
    a = UniqueSeq(x)
    assert len(a) == len(set(a))


# 1 is not iterable
# In [1, [1, 2, 3]], [1, 2, 3] is not hashable
@pytest.mark.parametrize("x", [1, [1, [1, 2, 3]]])
def test_new_failure(x) -> None:
    with pytest.raises(TypeError):
        _ = UniqueSeq(x)


# We block the concate operation from tuple. Since UniqueSeq need to keep
# uniqueness, if we implement concate operation with other iterable, we cannot
# gauranttee the length of the result equal the the sum of the length of the two
# iterables, and it can cause confusion and make the behavior unpredictable.
def test_add_failure() -> None:
    a = UniqueSeq([1, 2, 3])
    # UniqueSeq.__add__
    with pytest.raises(TypeError):
        _ = a + [1]


# Same reason with concate operation above, we block the mul operation.
def test_mul_failure() -> None:
    a = UniqueSeq([1, 2, 3])
    # UniqueSeq.__mul__
    with pytest.raises(TypeError):
        _ = a * 2
    # UniqueSeq.__rmul__
    with pytest.raises(TypeError):
        _ = 2 * a


# To make up the lack of add and mul operation, we introduce the or operation.
# This operation will have similar behavior with the set.__or__ operation, which
# will keep the uniqueness of the results. However UniqueSeq.__or__ will keep
# the order of the arguments, and it accepts any instanace of Iterable[Hashable]
# as the right operand.
@pytest.mark.parametrize("b", [[1, 2, 4], (1, 2, 4), iter([1, 2, 4])])
def test_or(b) -> None:
    a = UniqueSeq([1, 2, 3])
    # UniqueSeq.__or__
    c = a | b
    assert isinstance(c, UniqueSeq)
    assert c == (1, 2, 3, 4)


# We intentionally don't implement the UniqueSeq.__ror__ operation. It is
# aligned with the built-in collection behavior. For example
# >>> a = {1, 2, 3}
# >>> b = frozenset([1, 2, 4])
# >>> type(a | b)
# <class 'set'>
# >>> type(b | a)
# <class 'frozenset'>
# The result and the behavior depend on the left operand.
def test_or_failure() -> None:
    a = UniqueSeq([1, 2, 3])
    b = [1, 2, 4]
    with pytest.raises(TypeError):
        _ = b | a


# We also implement the UniqueSeq.union method, in case we want to combine
# multiple iterables.
def test_union() -> None:
    a = UniqueSeq([1, 2, 3])
    c = a.union([1, 2, 4], (1, 2, 5), iter((1, 2, 6)))
    assert isinstance(c, UniqueSeq)
    assert c == (1, 2, 3, 4, 5, 6)


# Other common sequence operations, these are inherited from the tuple class.
# https://docs.python.org/3/library/stdtypes.html#common-sequence-operations
def test_other_common_seq_operations() -> None:
    s = UniqueSeq(range(10))
    # x in s
    assert 5 in s
    # x not in s
    assert 10 not in s
    # s[i]
    assert s[5] == 5
    # s[i:j]
    assert s[3:6] == (3, 4, 5)
    # s[i:j:k]
    assert s[2:8:2] == (2, 4, 6)
    # len(s)
    assert len(s) == 10
    # min(s)
    assert min(s) == 0
    # max(s)
    assert max(s) == 9
    # s.index(x[, i[, j]])
    assert s.index(5) == 5
    # s.count(x)
    assert s.count(5) == 1


# repr of UniqueSeq
def test_repr() -> None:
    a = UniqueSeq([1, 2, 3])
    assert repr(a) == "UniqueSeq((1, 2, 3))"


# UniqueSeq contains a generic type parameter and support type hinting.
# UniqueSeq can only accept one item type, similar to Iterable.
def test_single_item_type() -> None:
    with pytest.raises(TypeError):
        UniqueSeq[int, str]


# UniqueSeq item type can be None and instances of typing.Type,
# types.GenericAlias and types.UnionType, as long as they are hashable.
# None: UniqueSeq[None] can actually only have one instance UniqueSeq((None,))
# int: is instance of typing.Type
# tuple[int, str]: is instance of types.GenericAlias
# int | str: is instance of types.UnionType
# tuple[int, str] | None: is instance of types.UnionType
@pytest.mark.parametrize(
    "T", [None, int, tuple[int, str], int | str, tuple[int, str] | None]
)
def test_item_type_success(T) -> None:
    UniqueSeq[T]


# UniqueSeq is also compatible with pydantic Models
# pydantic.BaseModel will automatically parse the input
@pytest.mark.parametrize(
    "a",
    [[1, 2, 3], (1, 2, 3), iter([1, 2, 3]), ["1", 2, "3"]],
)
def test_pydantic_base_model_with_item_type_success(a) -> None:
    class TestModel(BaseModel):
        a: UniqueSeq[int]

    model = TestModel(a=a)
    assert isinstance(model.a, UniqueSeq)
    assert model.a == (1, 2, 3)


# If the input item type does not match, pydantic will raise ValidationError
def test_pydantic_base_model_with_item_type_failure() -> None:
    class TestModel(BaseModel):
        a: UniqueSeq[int]

    # here tuple (1, 2) is not an instance of item type int
    with pytest.raises(ValidationError):
        _ = TestModel(a=[(1, 2), 1])


# When no item type is provided, pydantic will test if input is hashable
def test_pydantic_base_model_no_item_type() -> None:
    class TestModel(BaseModel):
        a: UniqueSeq

    model = TestModel(a=[1, 2, 3])
    assert isinstance(model.a, UniqueSeq)
    assert model.a == (1, 2, 3)

    # list [1, 2] is not hashable
    with pytest.raises(TypeError):
        _ = TestModel(a=[[1, 2]])

    # One exception for this is that if we provide tuple[int, ...] as the item
    # type, it will convert the list to tuple and no error will be raised.
    class OtherTestModel(BaseModel):
        a: UniqueSeq[tuple[int, ...]]

    model = OtherTestModel(a=[[1, 2]])
    assert isinstance(model.a, UniqueSeq)
    assert model.a == ((1, 2),)


# Since UniqueSeq is a subclass of tuple, pydantic will serialize and
# deserialize in the same way as tuple.
def test_pydantic_base_model_serialization() -> None:
    class TestModel(BaseModel):
        a: UniqueSeq[int]

    model = TestModel(a=[1, 2, 2, 3])

    # serialization
    model_dict = model.model_dump()
    assert model_dict == {"a": (1, 2, 3)}

    # deserialization
    other_model = TestModel(**model_dict)
    assert isinstance(other_model.a, UniqueSeq)
    assert other_model.a == (1, 2, 3)
