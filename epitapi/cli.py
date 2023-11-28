import fnmatch
import functools as ft
from pathlib import Path
from sys import argv
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
@click.pass_context
def cli(ctx=None, activity: str = "*"):
    assert ctx is not None
    if not any(c in activity for c in "*?[]"):
        ctx.obj = ApiActivity(activity)
        return


@cli.result_callback()
def check_and_run(actions: list):
    print(" | ".join(f"{a.__annotations__['name']}" for a in actions))
    dry_run = set()
    for a in actions:
        in_set = set(a.__annotations__["in"])
        if in_set > dry_run:
            print(f"{a.__annotations__['name']}: missing {dry_run - in_set}")
            print(f"had:{dry_run} need:{in_set}")
            return
        dry_run = dry_run.union(a.__annotations__["out"])

    ctx = {}
    for ac in actions:
        res = ac(**{x: ctx[x] for x in ac.__annotations__["in"]})
        if res is None:
            print(f"Interrupted on {ac.__annotations__['name']}")
            return
        if set(res.keys()) != set(ac.__annotations__["out"]):
            print(
                f"{ac.__annotations__['name']} broke out contract:\ngot:{res.keys()} expected:{ac.__annotations__['out']}"
            )
            return
        ctx.update(res)


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

    def _tmp() -> dict:
        return {"api": ApiActivity(activity, l_lst), "filters": list()}

    _tmp.__annotations__.update(
        {
            "name": "activity",
            "in": [],
            "out": ["api", "filters"],
        }
    )
    return _tmp


@cli.command()
def grps():
    """Display groups in an activity."""

    def _tmp(api: ApiActivity):
        print("\n".join([g.members[0] for g in api.groups.values()]))
        return {}

    _tmp.__annotations__.update(
        {
            "name": "grps",
            "in": ["api"],
            "out": [],
        }
    )
    return _tmp


@cli.command()
@click.argument("field", type=click.STRING)
def disp(field):
    """Display value from context."""

    def _tmp(**kwargs):
        print(kwargs[field])
        if isinstance(kwargs[field], dict) or isinstance(kwargs[field], list):
            print(len(kwargs[field]))
        return {}

    _tmp.__annotations__.update(
        {
            "name": "disp",
            "in": [field],
            "out": [],
        }
    )
    return _tmp


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

    def _tmp(api: ApiActivity, filters: list) -> dict | None:
        filters.extend(filts)
        if test:
            print(
                "\n".join(s for s in api.submissions_def if all(f(s) for f in filters))
            )
            return None
        return {"filters": filters}

    _tmp.__annotations__.update(
        {
            "name": "filter",
            "in": ["api", "filters"],
            "out": ["filters"],
        }
    )
    return _tmp


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
def grade(select, mark, merge):
    """Grades students for selected submissions (cf. filters)"""

    def _tmp(api: ApiActivity, filters: list[Callable[[str], bool]]) -> dict:
        grades = {}
        if api.students is not None:
            grades = {s: 0.0 for s in api.students}
        grades.update(grade_activity(api, filters, select, mark, merge))
        return {"grades": grades}

    _tmp.__annotations__.update(
        {
            "name": "grade",
            "in": ["filters", "api"],
            "out": ["grades"],
        }
    )
    return _tmp


@cli.command()
@click.argument("field", type=click.STRING)
def tocsv(field: str):
    def _tmp(api: ApiActivity, **kwargs) -> dict:
        with open(
            f"{api.activity.split('%2F')[1]}_{field}.csv", "w", encoding="utf-8"
        ) as f:
            f.writelines(f"{k},{v}\n" for k, v in kwargs[field].items())
        return {}

    _tmp.__annotations__.update(
        {
            "name": "tocsv",
            "in": ["api", field],
            "out": [],
        }
    )
    return _tmp


if __name__ == "__main__":
    cli()
