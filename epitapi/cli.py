import fnmatch
import functools as ft
import itertools as it
import json
from pathlib import Path
from subprocess import check_call
from typing import Callable, Optional

import click

from .api import ApiOperator
from .api_operator import ApiActivity
from .grades import grade_activity


@click.group(
    chain=True,
    invoke_without_command=True,
    no_args_is_help=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.version_option()
@click.option(
    "--update",
    is_flag=True,
    default=False,
    help="Reinstall epitapi with pipx",
)
@click.pass_context
def cli(ctx, update: bool = False):
    assert ctx is not None

    if update:
        check_call("pipx reinstall epitapi", shell=True)
        ctx.exit()
    return


def filter_args(x) -> set[str]:
    return set(a for a in x.__annotations__.keys() if a not in ("return", "kwargs"))


@cli.result_callback()
@click.pass_context
def check_and_run(ctx, actions: list, **kwargs):
    print(" | ".join(f"{a.__name__[1:]}({filter_args(a)})" for a in actions))

    ctx_vars = {}
    for ac in actions:
        for arg in filter_args(ac):
            if arg not in ctx_vars:
                click.echo(f"{arg} is missing from ctx (needed by {ac.__name__[1:]})")
                ctx.abort()
        res = ac(**{x: ctx_vars[x] for x in filter_args(ac)})
        if res is None:
            click.echo(f"Stopped on {ac.__name__[1:]}")
            ctx.abort()

        ctx_vars.update(res)


@cli.command()
@click.argument("activity", type=click.STRING)
@click.option(
    "-l",
    "--logins",
    type=click.File("r"),
    default=None,
    help="List of logins (one per line)",
)
@click.pass_context
def act(ctx: click.Context, activity: str, logins: Optional[click.File]):
    """The activity you are selecting. It usualy follows a tenant/activity pattern.

    Using globbing will list matching activities
    """
    if any(c in activity for c in "?*[]"):
        activity_list = ApiOperator().activities()
        click.echo(
            "\n".join(x for x in activity_list if fnmatch.fnmatch(x, f"*{activity}*"))
        )
        ctx.exit()

    l_lst = None
    if logins is not None:
        l_lst = Path(logins.name).read_text("utf-8").splitlines()

    def _activity() -> dict:
        return {"api": ApiActivity(activity, l_lst), "filters": list()}

    return _activity


@cli.command()
def grps():
    """Display groups in an activity."""

    def _grps(api: ApiActivity):
        print("\n".join([g.members[0] for g in api.groups.values()]))
        return {}

    return _grps


@cli.command()
@click.argument("field", type=click.STRING)
def disp(field):
    """Display value from context."""

    def _disp(**kwargs):
        print(json.dumps(kwargs[field], indent=2))
        if isinstance(kwargs[field], dict) or isinstance(kwargs[field], list):
            print(len(kwargs[field]))
        return {}

    _disp.__annotations__[field] = str
    return _disp


@cli.command("filter")
@click.option(
    "-f",
    "--filter",
    "pf",
    type=click.STRING,
    multiple=True,
    help="Add a positive filter",
)
@click.option(
    "-nf",
    "--neg-filter",
    "nf",
    type=click.STRING,
    multiple=True,
    help="Add a negative filter",
)
@click.option(
    "-t",
    "--test",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print everything matched by the filters",
)
def filt(pf, nf, test: bool):
    """Register filters for the submissions. All filters use globbing"""

    def not_fnmatch(name: str, pat: str) -> bool:
        return not fnmatch.fnmatch(name, pat)

    filts = [ft.partial(fnmatch.fnmatch, pat=pat) for pat in pf]
    filts.extend((ft.partial(not_fnmatch, pat=pat) for pat in nf))

    def _filter(api: ApiActivity, filters: list) -> dict | None:
        filters.extend(filts)
        if test:
            print(
                "\n".join(s for s in api.submissions_def if all(f(s) for f in filters))
            )
            return None
        return {"filters": filters}

    return _filter


@cli.command()
@click.option(
    "-s",
    "--select",
    type=click.Choice(grade_activity.__annotations__["select"].__args__),
    required=True,
    help="How to select for each assignments",
)
@click.option(
    "-ma",
    "--mark",
    type=click.Choice(grade_activity.__annotations__["mark"].__args__),
    required=True,
    help="How to mark a submission",
)
@click.option(
    "-me",
    "--merge",
    type=click.Choice(grade_activity.__annotations__["merge"].__args__),
    required=True,
    help="How to merge each assignment's mark",
)
@click.pass_context
def grade(ctx, select, mark, merge):
    """Grades students for selected submissions (cf. filters)"""

    def _grade(api: ApiActivity, filters: list[Callable[[str], bool]]) -> dict:
        if api.students is None:
            click.echo("Student list is required for grading. (see: act -l login.lst)")
            ctx.abort()
        grades = {}
        if api.students is not None:
            grades = {s: 0.0 for s in api.students}
        grades.update(grade_activity(api, filters, select, mark, merge))
        return {"grades": grades}

    return _grade


@cli.command()
def rerun():
    """Rerun selected traces"""

    def _rerun(api: ApiActivity, filters: list[Callable[[str], bool]]) -> dict:
        subs = map(
            lambda x: (x.submissionDefinitionUri, x.id),
            sorted(
                filter(
                    (lambda x: all(f(x.submissionDefinitionUri) for f in filters)),
                    api.explore(),
                ),
                key=lambda x: x.submissionDefinitionUri,
            ),
        )

        for subUri, grp in it.groupby(subs, key=lambda x: x[0]):
            ids = [x[1] for x in grp]
            if click.confirm(f"Rerun {len(ids)} for {subUri}", default=True):
                api.prepare(subUri, ids)
        return {}

    return _rerun


@cli.command()
@click.option(
    "-a",
    "--all",
    "is_all",
    is_flag=True,
    show_default=True,
    default=False,
    help="Publish all submissions and not just the last",
)
@click.argument("state", type=bool)
def publish(is_all, state: bool):
    """Bulk-(Un)Publish selected traces"""

    def _publish(api: ApiActivity, filters: list[Callable[[str], bool]]) -> dict:
        subUri = [x for x in api.submissions_def if all(f(x) for f in filters)]

        click.echo("\n".join(subUri))
        if click.confirm(
            f"Bulk {'' if state else 'un-'}publish {'all' if is_all else 'last'}",
            default=True,
        ):
            res = api.bulk_publish(
                subUri, "ALL" if is_all else "LAST_FOR_SUBMISSION_DEFINITION", state
            )
            click.echo(f"Publishing {len(res)}")
        return {}

    return _publish


@cli.command()
@click.argument("field", type=click.STRING)
def tocsv(field: str):
    """Save variable as csv file. Only grades is supported for now"""

    def _tocsv(api: ApiActivity, **kwargs) -> dict:
        with open(
            f"{api.activity.split('%2F')[1]}_{field}.csv", "w", encoding="utf-8"
        ) as f:
            f.writelines(f"{k},{v}\n" for k, v in kwargs[field].items())
        click.echo(f"Wrote to {api.activity.split('%2F')[1]}_{field}.csv")
        return {}

    _tocsv.__annotations__[field] = str
    return _tocsv


if __name__ == "__main__":
    cli()
