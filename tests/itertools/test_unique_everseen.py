from collections.abc import Iterator
from itertools import tee
from msca.itertools import unique_everseen
import pytest


@pytest.mark.parametrize(
    "iterable", ["abc", [1, 2, 3], (1, 2, 3), iter((1, 2, 3)), ("abc", "efg")]
)
def test_unique_everseen(iterable):
    if isinstance(iterable, Iterator):
        iterable, copy = tee(iterable)
    else:
        iterable, copy = iterable, iterable
    assert tuple(unique_everseen(iterable)) == tuple(dict.fromkeys(copy).keys())
