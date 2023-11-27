import itertools as it
from fnmatch import fnmatch
from typing import Callable, Iterable, Literal, Tuple

from . import datatypes as dt
from .api_operator import ApiActivity


def select_last(
    s: Iterable[Tuple[dt.Submission, float]]
) -> Tuple[dt.Submission, float]:
    return max(s, key=lambda x: x[0].receivedAt)


def select_best(
    s: Iterable[Tuple[dt.Submission, float]]
) -> Tuple[dt.Submission, float]:
    return max(s, key=lambda x: x[1])


def mark_binary(x: dt.Submission) -> float:
    return x.validated


def mark_percent(x: dt.Submission) -> float:
    assert x.currentJob is not None
    assert x.currentJob.successPercent is not None
    return x.currentJob.successPercent / 100


def merge_avg(count: int):
    def _merge_avg(x: Iterable[Tuple[dt.Submission, float]]) -> float:
        return sum(y[1] for y in x) / count * 20

    return _merge_avg


def merge_last(x: Iterable[Tuple[dt.Submission, float]]) -> float:
    return max(x, key=lambda x: x[0].receivedAt)[1] * 20


def sorted_group_by(i, key):
    return it.groupby(sorted(i, key=key), key=key)


def grade_activity(
    act: ApiActivity,
    match: str,
    select: Literal["best", "last"],
    mark: Literal["binary", "percent"],
    merge: Literal["last", "avg"],
) -> dict[str, float]:
    total = sum(1 for x in act.submissions_def if fnmatch(x, match))

    select_fn: dict[
        str,
        Callable[[Iterable[Tuple[dt.Submission, float]]], Tuple[dt.Submission, float]],
    ] = {
        "best": select_best,
        "last": select_last,
    }

    mark_fn: dict[
        str,
        Callable[[dt.Submission], float],
    ] = {
        "binary": mark_binary,
        "percent": mark_percent,
    }

    merge_fn: dict[
        str,
        Callable[[Iterable[Tuple[dt.Submission, float]]], float],
    ] = {
        "last": merge_last,
        "avg": merge_avg(total),
    }

    subs = (
        (x, mark_fn[mark](x))
        for x in act.explore(submissionStatus="SUCCEEDED")
        if True or fnmatch(x.submissionDefinitionUri, match)
    )

    return {
        act.groups[slug].members[0]: merge_fn[merge](
            (
                select_fn[select](sub_grp)
                for _, sub_grp in sorted_group_by(
                    stud_grp, key=lambda x: x[0].submissionDefinitionUri
                )
            ),
        )
        for slug, stud_grp in sorted_group_by(subs, key=lambda x: x[0].groupSlug)
    }
