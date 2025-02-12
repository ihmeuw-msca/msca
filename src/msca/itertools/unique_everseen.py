import itertools
from collections.abc import Hashable, Iterable, Iterator


def unique_everseen[T: Hashable](
    iterable: Iterable[T], seen: Iterable[T] = ()
) -> Iterator[T]:
    seen = set(seen)
    for item in itertools.filterfalse(seen.__contains__, iterable):
        seen.add(item)
        yield item
