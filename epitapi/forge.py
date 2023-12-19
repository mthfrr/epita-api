#!/usr/bin/env python3
import contextlib
import fnmatch
import inspect
import itertools as it
import json
import logging
import math
import subprocess as sp
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta
from functools import reduce, wraps
from http.client import HTTPConnection
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
from numpy.typing import ArrayLike
from pydantic import BaseModel


def log_requests(func):
    @wraps(func)
    @contextlib.contextmanager
    def wrapper_func(*args, **kwargs):
        HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True
        res = func(*args, **kwargs)
        HTTPConnection.debuglevel = 0
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        root_logger.handlers = []
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.WARNING)
        requests_log.propagate = False
        return res

    return wrapper_func


class Job(BaseModel):
    id: str
    status: Literal[
        "FAILED_PREPARATION",
        "IDLE",
        "PREPARED",
        "PREPARING",
        "RESULT_UPLOADED",
        "RUNNER_DISPATCHED",
        "RUNNER_DISPATCH_FAILED",
        "RUNNER_FAILED",
        "RUNNER_SUCCEEDED",
        "RUNNER_WAITING",
        "UPLOAD_FAILED",
    ]
    successPercent: int | None
    maasPipelineId: str | None
    maasExternalId: str | None
    maasSubmissionUrl: str | None
    testSuiteUrl: str | None
    traceUrl: str | None
    bundleUrl: str | None
    retryCount: int
    retryLast: datetime
    retryLocked: bool
    isCurrent: bool
    version: int
    runtimeClass: str


class Submission(BaseModel):
    id: str
    status: Literal["ERROR", "IDLE", "PROCESSING", "SUCCEEDED"]
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
    ] | None
    platform: Literal["FORMS", "LEAGUE", "MAAS"]
    ref: str
    repository: str
    groupSlug: str
    ip: str
    author: str
    receivedAt: datetime
    published: bool
    validated: bool
    activityUri: str
    assignmentUri: str
    submissionDefinitionUri: str
    jobs: list[Job]
    currentJob: Job | None


class TraceExplorerResponse(BaseModel):
    count: int
    pageCount: int
    results: list[Submission]


class SimpleGroup(BaseModel):
    uri: str
    slug: str
    members: list[str]
    gitRemoteUrl: str


class Group(BaseModel):
    uri: str
    slug: str
    members: list[str]
    submissions: list[str]
    gitRemoteUrl: str


class Forge:
    api_url = "https://operator.forge.epita.fr/api"

    def __init__(self, activity: str):
        self.session = requests.Session()
        self.activity = urllib.parse.quote_plus(activity)
        self.refresh_token()

        self._groups = self.get_groups()
        self._assignment = self.get_assignment()
        self._submissionDefinitionUri = self.get_submissionDefinitionUri()

    def refresh_token(self):
        token = sp.check_output(["poulpy", "auth"]).decode().strip()
        self.session.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    @staticmethod
    def auto_refresh_token(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                res = func(self, *args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if e.response is None:
                    raise e
                if e.response.status_code == 401:
                    self.refresh_token()
                return func(self, *args, **kwargs)
            return res

        return wrapper

    @auto_refresh_token
    def _traces_explore(
        self, page: int, size: int = 50, **kwargs
    ) -> TraceExplorerResponse:
        params = {"pageNum": page, "pageSize": size, **kwargs}
        params = {x[0]: x[1] for x in params.items() if x[1] is not None}
        res = self.session.get(
            f"{self.api_url}/traces/{self.activity}/explore", params=params
        )
        res.raise_for_status()
        return TraceExplorerResponse.model_validate_json(res.content)

    def trace_explore(
        self,
        filtered: bool = True,
        pageSize: int = 100,
        groupSlug: str | None = None,
        submissionDefinitionUri: str | None = None,
        submissionStatus: Literal["ERROR", "IDLE", "PROCESSING", "SUCCEEDED"]
        | None = None,
        published: bool | None = None,
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
    ) -> TraceExplorerResponse:
        explo_resp = self._traces_explore(
            0,
            pageSize,
            groupSlug=groupSlug,
            submissionDefinitionUri=submissionDefinitionUri,
            submissionStatus=submissionStatus,
            published=published,
            error=error,
        )
        page_count = explo_resp.pageCount
        if page_count > 1:
            with ThreadPool(page_count - 1) as p:
                explo = p.map(
                    lambda x: self._traces_explore(
                        x,
                        pageSize,
                        groupSlug=groupSlug,
                        submissionDefinitionUri=submissionDefinitionUri,
                        submissionStatus=submissionStatus,
                        published=published,
                        error=error,
                    ),
                    range(1, page_count),
                    1,
                )
            explo_resp.results.extend(it.chain(*[x.results for x in explo]))

        if filtered:
            assign = self.assignment
            grps = self.groups
            r = filter(lambda x: x.assignmentUri in assign, explo_resp.results)
            r = filter(lambda x: x.groupSlug in grps, r)
            explo_resp.results = [*r]
            explo_resp.count = len(explo_resp.results)

        return explo_resp

    @auto_refresh_token
    def get_result(self, job_id: str) -> str:
        res = self.session.get(
            f"{self.api_url}/traces/{self.activity}/job/{job_id}/maas-result"
        )
        res.raise_for_status()
        return res.text

    @property
    def groups(self) -> dict[str, SimpleGroup]:
        if not hasattr(self, "_groups"):
            self._groups = self.get_groups()
        return self._groups

    @auto_refresh_token
    def get_groups(self) -> dict[str, SimpleGroup]:
        res = self.session.get(
            f"{self.api_url}/activities/groups/{self.activity}",
        )
        res.raise_for_status()
        return {x.slug: x for x in (SimpleGroup.model_validate(x) for x in res.json())}

    @property
    def assignment(self) -> list[str]:
        if not hasattr(self, "_assignment"):
            self._assignment = self.get_assignment()
        return self._assignment

    @auto_refresh_token
    def get_assignment(self) -> list[str]:
        res = self.session.get(
            f"{self.api_url}/activities/activity/{self.activity}/my-assignemnts",
        )
        res.raise_for_status()
        return res.json()

    @property
    def submissionDefinitionUri(self) -> list[str]:
        if not hasattr(self, "_submissionDefinitionUri"):
            self._submissionDefinitionUri = self.get_submissionDefinitionUri()
        return self._submissionDefinitionUri

    @auto_refresh_token
    def get_submissionDefinitionUri(self) -> list[str]:
        res = self.session.get(
            f"{self.api_url}/activities/activity/{self.activity}/my-submissions-definitions",
        )
        res.raise_for_status()
        return res.json()

    @auto_refresh_token
    def _bulk_publish(
        self,
        submissionDefinitionUri: str,
        pick: Literal["ALL", "LAST_FOR_SUBMISSION_DEFINITION"],
        published: bool = True,
    ) -> list[str]:
        res = self.session.put(
            f"{self.api_url}/traces/{urllib.parse.quote_plus(submissionDefinitionUri)}/submission/bulk-publish",
            params={"pick": pick, "published": published},
        )
        res.raise_for_status()
        return res.json()

    @auto_refresh_token
    def bulk_publish(
        self,
        submissionDefinitionUris: list[str],
        pick: Literal["ALL", "LAST_FOR_SUBMISSION_DEFINITION"] = "ALL",
        published: bool = True,
    ) -> list[str]:
        with ThreadPool(len(submissionDefinitionUris)) as p:
            res = p.map(
                lambda x: self._bulk_publish(x, pick=pick, published=published),
                submissionDefinitionUris,
                1,
            )
        return [*it.chain.from_iterable(res)]

    @auto_refresh_token
    def job_retry(self, submissionDefinitionUri: str, ids: list[str]):
        res = self.session.put(
            f"{self.api_url}/traces/{urllib.parse.quote_plus(submissionDefinitionUri)}/job/retry",
            json=ids,
        )
        res.raise_for_status()

    @auto_refresh_token
    def prepare(self, submissionDefinitionUri: str, ids: list[str]):
        res = self.session.put(
            f"{self.api_url}/traces/{urllib.parse.quote_plus(submissionDefinitionUri)}/submission/prepare",
            json=ids,
        )
        res.raise_for_status()


def publish_all(activityUri: str):
    f = Forge(activityUri)
    sub = f.get_submissionDefinitionUri()
    return f.bulk_publish(sub)


def unpublish_all(activityUri: str):
    f = Forge(activityUri)
    sub = f.get_submissionDefinitionUri()
    f.bulk_publish(sub, published=False)
    return


def get_failed(activityUri: str) -> list[Submission]:
    f = Forge(activityUri)
    submissions = f.trace_explore()
    return [
        x
        for x in submissions.results
        if (
            x.status == "ERROR"
            and (
                x.jobs != []
                and (
                    (
                        x.currentJob is not None
                        and x.currentJob.status == "RUNNER_DISPATCH_FAILED"
                    )
                    or (
                        x.currentJob is None
                        and x.jobs[0].status == "RUNNER_DISPATCH_FAILED"
                    )
                )
            )
            or x.error in ("MAX_IN_FLIGHT_EXCEEDED", "QUOTA_EXCEEDED")
        )
    ]


def get_nocurrent(activityUri: str) -> list[Submission]:
    f = Forge(activityUri)
    submissions = f.trace_explore()
    return [
        x
        for x in submissions.results
        if x.currentJob is None and x.status not in ("IDLE", "PROCESSING")
    ]


def get_processing(activityUri: str) -> list[Submission]:
    f = Forge(activityUri)
    submissions = f.trace_explore()
    return [
        x
        for x in submissions.results
        if x.currentJob is None and x.status in ("IDLE", "PROCESSING")
    ]


def rerun_failed(activityUri: str):
    f = Forge(activityUri)

    d = defaultdict(list)
    for s in f.trace_explore(error="JOB_ERROR").results:
        d[s.submissionDefinitionUri].append(s.id)
        # print(
        #     f"{s.assignmentUri.split('/')[-1]:<10}\t{f.get_groups()[s.groupSlug].members[0]}\t{s.ref}"
        # )

    for k, v in d.items():
        # print(f"{k.split('/')[-1]} {len(v)}")
        f.prepare(k, v)


def rerun_exo(activityUri: str):
    f = Forge(activityUri)

    submissions = f.trace_explore(
        submissionDefinitionUri="epita-bachelor-cyber/prog-c-s1-2026/root/exercises/hello_world/hello_world/hello_world"
    )
    print(len(submissions.results))
    f.prepare(
        "epita-bachelor-cyber/prog-c-s1-2026/root/exercises/hello_world/hello_world/hello_world",
        [x.id for x in submissions.results],
    )


def find_progress(activityUri: str):
    f = Forge(activityUri)
    submissions = f.trace_explore()

    grps = f.get_groups()

    print(grps)

    todo = [
        "hello_world",
        "my_round",
        "my_pow",
        "my_abs",
        # "prime_factorization",
        "digit",
        "int_palindrome",
        "number_digits",
    ]

    out: dict[str, int] = {}

    for g in grps:
        res: dict[str, bool] = {x: False for x in todo}
        for x in (y for y in submissions.results if y.groupSlug == g):
            name = x.assignmentUri.split("/")[-1]
            if name in todo:
                res[name] = res[name] or x.validated

        out[grps[g].members[0]] = sum(res.values())

    for i in range(len(todo) + 1):
        studs = [x for x in out.items() if x[1] == i]
        print(f"\n### {i} -> {len(studs)} students")
        for s in studs:
            print(s[0])

    print(len(grps))


def save_hist(
    grades: ArrayLike,
    date: datetime,
    path: Path = Path("./hists"),
):
    path.mkdir(exist_ok=True)
    hist = np.unique(grades, return_counts=True)

    plt.bar(hist[0], hist[1], width=1.0, align="edge")
    plt.title(f"CYB1 - PROG-C Progress - {date.strftime('%d/%m/%Y %H:%M')}")
    plt.xticks(range(max(hist[0]) + 1))
    plt.yticks(range(max(hist[1]) + 1))
    # plt.show()
    plt.savefig(path / (date.strftime("%Y%m%d_%H_%M") + ".png"))
    plt.clf()
    plt.cla()


def get_advancement_at(
    subms: list[Submission], grps: dict[str, SimpleGroup], date: datetime
) -> dict[str, int]:
    out: dict[str, int] = {}

    for g in grps:
        res: dict[str, bool] = defaultdict(lambda: False)
        for x in subms:
            if x.groupSlug == g and x.receivedAt < date:
                res[x.assignmentUri] = res[x.assignmentUri] or x.validated

        out[grps[g].members[0]] = sum(res.values())

    return out


def stat(activityUri: str):
    f = Forge(activityUri)
    submissions = f.trace_explore()
    grps = f.groups
    del grps["epita-bachelor-cyber-prog-c-s1-2026-valentin.seux"]

    curr_date = pytz.timezone("Europe/Paris").localize(
        datetime.strptime("2023-11-07", "%Y-%m-%d")
    )
    now = pytz.timezone("Europe/Paris").localize(datetime.now())

    out = get_advancement_at(
        submissions.results,
        grps,
        now,
    )
    print("\n".join(f"{x[0]}: {x[1]}" for x in sorted(out.items(), key=lambda x: x[1])))
    df = pd.DataFrame(out.values())
    print(df.describe())
    save_hist(list(out.values()), date=now)

    while curr_date < now:
        out = get_advancement_at(
            submissions.results,
            grps,
            curr_date,
        )
        save_hist(list(out.values()), date=curr_date)
        curr_date += timedelta(days=0, hours=2)


def run_last_sub(activityUri: str):
    f = Forge(activityUri)
    subs = f.trace_explore()

    ids = []
    for s in subs.results:
        ids.append(s.id)

    f.prepare(subs.results[0].submissionDefinitionUri, ids)


def disp_hist(grades: Iterable[Any]):
    hist = np.unique(np.array([int(x) for x in grades]), return_counts=True)
    plt.bar(hist[0], hist[1], width=1.0, align="edge")
    plt.xticks(range(max(hist[0] + 1)))
    plt.yticks(range(max(hist[1] + 1)))
    plt.show()


def status(activityUri: str):
    f = Forge(activityUri)

    cats: dict[str, int] = defaultdict(lambda: 0)
    for sub in f.trace_explore().results:
        cats[sub.status] += 1
    print(json.dumps(cats, indent=2))


def last(s: Iterable[Submission]) -> list[Submission]:
    res = []
    for _, grp in it.groupby(
        sorted(s, key=lambda x: x.assignmentUri), lambda x: x.assignmentUri
    ):
        res.append(max(grp, key=lambda x: x.receivedAt))

    return res


def best(s: Iterable[Submission]) -> list[Submission]:
    res = []
    for _, grp in it.groupby(
        sorted(s, key=lambda x: x.assignmentUri), lambda x: x.assignmentUri
    ):
        res.append(max(grp, key=lambda x: x.currentJob.successPercent, default=0))
    return res


def mark_percent(s: Iterable[Submission], count: int) -> float:
    m = 0.0
    for x in s:
        assert x.currentJob is not None
        assert x.currentJob.successPercent is not None
        m += x.currentJob.successPercent / 100

    return m / count * 20


def mark_binary(s: Iterable[Submission], count: int) -> float:
    m = 0.0
    for x in s:
        m += x.validated

    return m / count * 20


def compute_grades(
    activityUri: str,
    select: Callable[[Iterable[Submission]], Iterable[Submission]],
    mark: Callable[[Iterable[Submission], int], float],
    match: str = "*exercise*",
) -> dict[str, float]:
    f = Forge(activityUri)
    subs = f.trace_explore(submissionStatus="SUCCEEDED")

    res: dict[str, float] = {x[1].members[0]: 0.0 for x in f.groups.items()}
    subUri = [x for x in f.submissionDefinitionUri if fnmatch.fnmatch(x, match)]

    print(subUri)

    count = len(subUri)
    print(f"{len(f.submissionDefinitionUri)} -> {count}")

    for slug, stud_subs in it.groupby(
        sorted(
            (x for x in subs.results if x.submissionDefinitionUri in subUri),
            key=lambda x: x.groupSlug,
        ),
        (lambda x: x.groupSlug),
    ):
        s = select(x for x in stud_subs if x.submissionDefinitionUri in subUri)
        m = mark(s, count)
        res[f.groups[slug].members[0]] = m

    res = {
        k: v
        for (k, v) in res.items()
        if k
        not in (
            "valentin.seux",
            "mathieu.fourre",
            "paul.lege",
            "thibault.viennot",
        )
    }

    with open(f"{activityUri.split('/')[-1]}.csv", "w", encoding="utf-8") as f:
        f.write(f"# select: {select.__name__} / grade: {mark.__name__}\n")
        f.writelines((f"{x[0]},{x[1]}\n" for x in res.items()))
    return res


def publish_dementor(activityUri: str, dementor: str):
    f = Forge(activityUri)
    print(f"{f.assignment[0]}/{dementor}")
    ids = f.bulk_publish(
        [f"{f.assignment[0]}/{dementor}"],
        pick="LAST_FOR_SUBMISSION_DEFINITION",
        published=True,
    )
    print(len(ids))


if __name__ == "__main__":
    print("> Main")
    # grades("epita-bachelor-cyber/2026-python")
    # print(publish_all("epita-bachelor-cyber/2026-python"))
    # rerun_failed("epita-bachelor-cyber/2026-python")
    # print(len(get_nocurrent("epita-bachelor-cyber/2026-python")))
    # print(get_processing("epita-bachelor-cyber/2026-python"))
    # rerun_exo("epita-bachelor-cyber/prog-c-s1-2026")
    # rerun_failed("epita-bachelor-cyber/prog-c-s1-2026")
    # find_progress("epita-bachelor-cyber/prog-c-s1-2026")

    # stat("epita-bachelor-cyber/prog-c-s1-2026")

    # run_last_sub("epita-bachelor-cyber/rle-python-2026")
    # status("epita-bachelor-cyber/rle-python-2026")
    # get_grades("epita-bachelor-cyber/rle-python-2026")
    # print(publish_all("epita-bachelor-cyber/rle-python-2026"))
    # g = compute_grades("epita-bachelor-cyber/rle-python-2026", last, mark_percent)
    # g = compute_grades("epita-bachelor-cyber/2026-python", best, mark_binary)
    # g = compute_grades("epita-bachelor-cyber/prog-c-s1-2026", best, mark_binary)
    g = compute_grades(
        "epita-bachelor-cyber/prog-c-tinyprintf-2026",
        last,
        mark_percent,
        match="*dementor2",
    )
    # print(len(g))
    # print(json.dumps(g, indent=2, sort_keys=True))

    disp_hist(list(g.values()))

    # publish_dementor(
    #     "epita-bachelor-cyber/prog-c-tinyprintf-2026", "tinyprintf-dementor1"
    # )
    # publish_dementor(
    #     "epita-bachelor-cyber/prog-c-tinyprintf-2026", "tinyprintf-dementor2"
    # )
