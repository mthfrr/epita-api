import functools as ft
import itertools as it
from fnmatch import fnmatch
from multiprocessing.pool import ThreadPool
from pathlib import Path
from runpy import run_path
from statistics import mean, median
from typing import Literal, TextIO, final

import click
from pydantic import BaseModel

from epitapi.gradeapi import Grader

from .api_operator import ApiActivity
from .sorted_group_by import sorted_group_by


class ConfActivity(BaseModel):
    ignore: list[str]
    grades: Literal["binary"]
    submissions: Literal["best", "last"]
    assignments: Literal["avg", "last"]


class ConfFile(BaseModel):
    activities: dict[str, ConfActivity]
    logins: list[str]


@click.group(
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "-f",
    "--file",
    type=click.File("r", "utf-8"),
    help="File the grading setting (default: grading.py)",
)
@click.argument("activity", required=False, default="", type=click.STRING)
@click.argument("login", required=False, default="", type=click.STRING)
@click.version_option()
def epi_grades(file: None | TextIO = None, activity: str = "", login: str = ""):
    if file is None:
        filename = "./grading.py"
    else:
        filename = file.name
        file.close()

    login_contains = login

    mod = run_path(filename)
    conf: dict[str, Grader] = mod["conf"]

    final_output = {}

    for act, c in conf.items():
        if activity not in act:
            click.echo(f"SKIP: {act}")
            continue

        click.echo(f"GRAD: {act}")
        a = ApiActivity(act)

        filt_grp = None
        if login_contains:
            filt_grp = [
                k
                for k, v in a.groups.items()
                if any(login_contains in login for login in v.members)
            ][0]

        assignments = [x for x in a.assignments if all(s not in x for s in c.ignore)]
        total = len(assignments)
        # click.echo(f"FILT: removed {set(a.assignments) - set(assignments)}")
        tags = [
            x
            for x in a.explore(submissionStatus="SUCCEEDED", groupSlug=filt_grp)
            if x.assignmentUri in assignments
            and all((y not in x.submissionDefinitionUri) for y in c.ignore)
        ]

        with ThreadPool(len(tags)) as p:
            gtags = p.map(c.per_tag, tags, 1)

        grp_grades = {
            slug: round(
                sum(
                    el[1]
                    for el in (
                        c.per_ass(
                            (
                                c.per_sub(sub_grp)
                                for _, sub_grp in sorted_group_by(
                                    ass_grp, lambda x: x[0].submissionDefinitionUri
                                )
                            )
                        )
                        for _, ass_grp in sorted_group_by(
                            slug_grp, lambda x: x[0].assignmentUri
                        )
                    )
                )
                / total
                * 20,
                2,
            )
            for slug, slug_grp in sorted_group_by(gtags, key=lambda x: x[0].groupSlug)
        }

        if login_contains:
            res = {}
        else:
            res = {login: 0.0 for login in mod["logins"]}
        for slug, grade in grp_grades.items():
            for login in a.groups[slug].members:
                assert grade >= 0 and grade <= 20
                res[login] = grade

        final_output[act] = res

        grades = list(res.values())
        click.echo(
            f"min:{min(grades):5.2f} mean:{mean(grades):5.2f} median:{median(grades):5.2f} max:{max(grades):5.2f}"
        )

    with open("./grades.tsv", "w", encoding="utf-8") as f:
        # acts = [*final_output.keys()]
        acts = [
            "epita-bachelor-cyber/2026-python",
            "epita-bachelor-cyber/rle-python-2026",
            "epita-bachelor-cyber/prog-c-s1-2026",
            "epita-bachelor-cyber/prog-c-tinyprintf-2026",
            "epita-bachelor-cyber/prog-c-rle-2026",
            "epita-bachelor-cyber/prog-c-minils-2026",
        ]

        f.write("logins\t")
        f.write("\t".join(x.split("/")[1] for x in acts))
        f.write("\n")

        for login in mod["logins"]:
            f.write(f"{login}\t")
            f.write("\t".join(f"{final_output[act][login]:.2f}" for act in acts))
            f.write("\n")


if __name__ == "__main__":
    epi_grades()
