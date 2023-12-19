from itertools import groupby
from typing import Any, Callable, Iterable, Iterator, Protocol, TypeVar

_T = TypeVar("_T")


class SupportsRichComparison(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...

    def __gt__(self, __other: Any) -> bool:
        ...


_U = TypeVar("_U", bound=SupportsRichComparison)


def sorted_group_by(
    i: Iterable[_T], key: Callable[[_T], _U]
) -> Iterable[tuple[_U, Iterator[_T]]]:
    return groupby(sorted(i, key=key), key=key)
