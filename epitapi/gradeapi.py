from functools import partial
from statistics import mean
from typing import Callable, Iterable

from pydantic import BaseModel

from epitapi.api_operator import ApiActivity

from . import datatypes as types

graded_submission = tuple[types.Submission, float]


class Grader(BaseModel):
    ignore: list[str]
    per_tag: Callable[[types.Submission], graded_submission]
    per_sub: Callable[[Iterable[graded_submission]], graded_submission]
    per_ass: Callable[[Iterable[graded_submission]], graded_submission]


def grade_binary(s: types.Submission) -> graded_submission:
    return (s, s.validated * 1.0)


def grade_raw_percent(s: types.Submission) -> graded_submission:
    assert s.currentJob is not None
    assert s.currentJob.successPercent is not None
    return (s, s.currentJob.successPercent / 100)


def grade_advanced(s: types.Submission) -> graded_submission:
    trace = ApiActivity.get_trace(s)

    # print(f"{s.groupSlug} {s.ref}: {trace.keys()}")

    if mean(x[1] for x in trace["build"]) != 1.0 or "Forbidden" in trace:
        return (s, 0.0)

    res = {k: sum(x[1] for x in v) / len(v) for k, v in trace.items()}
    assert set(trace.keys()) < set(
        (
            "build",
            "clang-format",
            "clang-tidy",
            "tests",
            "trashfiles",
            "asan",
        )
    )

    grade = res["tests"]
    # if "asan" in res:
    #     print("asan")
    #     grade -= len(trace["asan"]) / len(trace["tests"])

    malus = {
        "clang-format": 0.05,
        "clang-tidy": 0.05,
        "trashfiles": 0.1,
    }

    grade -= sum((1 - res[k]) * v for k, v in malus.items() if k in res)

    # print(f"{s.submissionDefinitionUri}")
    # print(f"{s.groupSlug} -> {grade}")
    return (s, max(0.0, grade))


def select_last(els: Iterable[graded_submission]) -> graded_submission:
    return max(els, key=lambda x: x[0].receivedAt)


def select_best(els: Iterable[graded_submission]) -> graded_submission:
    return max(els, key=lambda x: x[1])
