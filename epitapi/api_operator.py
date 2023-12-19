import itertools as it
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from fnmatch import fnmatch
from multiprocessing.pool import ThreadPool
from typing import Any, Iterable, Iterator, Literal, Protocol, TypeVar

import requests

from . import datatypes
from .api import ApiOperator

_API = ApiOperator()


class ApiActivity:
    api_url = ApiOperator.api_url
    activities = _API.activities()

    def __init__(self, activity: str, students: list[str] | None = None):
        self.api = _API
        if activity not in self.api.activities():
            raise KeyError(f"Activity {activity} not found")
        self.activity = urllib.parse.quote_plus(activity)
        self.students = students

        self.groups = self._groups()
        self.assignments = self._assignments()
        self.submissions_def = self._submissions_def()

    def _groups(self) -> dict[str, datatypes.SimpleGroup]:
        res = self.api.get(f"{self.api_url}/activities/groups/{self.activity}")
        return {
            x.slug: x
            for x in (datatypes.SimpleGroup.model_validate(x) for x in res.json())
        }

    def _assignments(self) -> list[str]:
        res = self.api.get(
            f"{self.api_url}/activities/activity/{self.activity}/my-assignemnts",
        )
        return res.json()

    def _submissions_def(self) -> list[str]:
        res = self.api.get(
            f"{self.api_url}/activities/activity/{self.activity}/my-submissions-definitions",
        )
        return res.json()

    def job_retry(self, submission_def: str, ids: list[str]) -> None:
        self.api.put(
            f"{self.api_url}/traces/{urllib.parse.quote_plus(submission_def)}/job/retry",
            json=ids,
        )

    def prepare(self, submission_def: str, ids: list[str]) -> None:
        self.api.put(
            f"{self.api_url}/traces/{urllib.parse.quote_plus(submission_def)}/submission/prepare",
            json=ids,
        )

    def bulk_publish(
        self,
        submissions_def: list[str],
        pick: Literal["ALL", "LAST_FOR_SUBMISSION_DEFINITION"] = "ALL",
        published: bool = True,
    ) -> list[str]:
        with ThreadPool(len(submissions_def)) as p:
            res = p.map(
                lambda x: self.api.put(
                    f"{self.api_url}/traces/{urllib.parse.quote_plus(x)}/submission/bulk-publish",
                    params={"pick": pick, "published": published},
                ).json(),
                submissions_def,
                1,
            )
        return [*it.chain.from_iterable(res)]

    def _explore(
        self, page: int, size: int = 50, **kwargs
    ) -> datatypes.TraceExplorerResponse:
        params = {"pageNum": page, "pageSize": size, **kwargs}
        params = {x[0]: x[1] for x in params.items() if x[1] is not None}
        res = self.api.get(
            f"{self.api_url}/traces/{self.activity}/explore", params=params
        )
        return datatypes.TraceExplorerResponse.model_validate_json(res.content)

    def explore(
        self,
        pageSize: int = 100,
        assignmentUri: str | None = None,
        error: Literal[
            "CLONE_FAILED",
            "DUPLICATED",
            "INVALID_GIT_REPOSITORY",
            "JOB_ERROR",
            "MAX_IN_FLIGHT_EXCEEDED",
            "NOT_ACCESSIBLE",
            "QUOTA_EXCEEDED",
            "REPOSITORY_TOO_BIG",
            "STUDENT_TAR_UPLOAD_FAILED",
            "TARBALL_CREATION_FAILED",
            "TESTSUITE_NOT_FOUND",
        ]
        | None = None,
        groupSlug: str | None = None,
        submissionDefinitionUri: str | None = None,
        submissionStatus: Literal["ERROR", "IDLE", "PROCESSING", "SUCCEEDED"]
        | None = None,
        validated: bool | None = None,
    ) -> list[datatypes.Submission]:
        info = self._explore(
            0,
            pageSize,
            assignmentUri=assignmentUri,
            error=error,
            groupSlug=groupSlug,
            submissionDefinitionUri=submissionDefinitionUri,
            submissionStatus=submissionStatus,
            validated=validated,
        )
        page_count = info.pageCount
        if page_count > 1:
            with ThreadPool(page_count - 1) as p:
                explo = p.map(
                    lambda x: self._explore(
                        x,
                        pageSize,
                        assignmentUri=assignmentUri,
                        error=error,
                        groupSlug=groupSlug,
                        submissionDefinitionUri=submissionDefinitionUri,
                        submissionStatus=submissionStatus,
                        validated=validated,
                    ),
                    range(1, page_count),
                    1,
                )
            info.results.extend(it.chain(*[x.results for x in explo]))

        r = filter(lambda x: x.groupSlug in self.groups, info.results)
        r = filter(lambda x: x.assignmentUri in self.assignments, r)
        if self.students is not None:
            r = filter(
                lambda x: all(
                    m in (self.students or []) for m in self.groups[x.groupSlug].members
                ),
                r,
            )
        return list(r)

    def submissions_as_dict(
        self, ignore: list[str]
    ) -> dict[str, dict[str, dict[str, list[datatypes.Submission]]]]:
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
            return it.groupby(sorted(i, key=key), key=key)

        return {
            grp: {
                assign: {
                    suburi: [*sub]
                    for suburi, sub in sorted_group_by(
                        subs, lambda x: x.submissionDefinitionUri
                    )
                    if not any(fnmatch(suburi, filt) for filt in ignore)
                }
                for assign, subs in sorted_group_by(all_sub, lambda x: x.assignmentUri)
            }
            for grp, all_sub in sorted_group_by(
                self.explore(submissionStatus="SUCCEEDED"), lambda x: x.groupSlug
            )
        }

    @classmethod
    def get_trace(cls, s: datatypes.Submission):
        assert s.currentJob is not None
        r = _API.put(
            f"{cls.api_url}/traces/{urllib.parse.quote_plus(s.submissionDefinitionUri)}/job/traces-urls",
            json=[s.currentJob.id],
            timeout=1000,
        )
        r = requests.get(r.json()[s.currentJob.id], timeout=1000)

        root = ET.fromstring(r.content)

        out = defaultdict(list)
        for el in root.findall(".//testcase"):
            path = el.attrib["classname"].split(".")
            out[path[1]].append(
                (
                    ".".join((*path[2:], el.attrib["name"])),
                    el.findall(".//failure") == [],
                )
            )
        return out
