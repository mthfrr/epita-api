from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class SimpleActivity(BaseModel):
    uri: str
    slug: str
    tenantSlug: str
    name: str
    version: int
    versionHash: str


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
    members: list[str]  # group logins
    gitRemoteUrl: str


class Group(BaseModel):
    uri: str
    slug: str
    members: list[str]  # list of logins
    submissions: list[str]  # list of submission ids
    gitRemoteUrl: str
