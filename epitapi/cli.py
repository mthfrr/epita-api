import fnmatch
from pathlib import Path
from sys import argv
from typing import Optional

import click
import pkg_resources

from .api import ApiOperator
from .api_operator import ApiActivity
from .grades import grade_activity


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.option("-v", "--version", is_flag=True, default=False, help="Print version")
def cli(version: bool = False):
    if version:
        info = pkg_resources.get_distribution("epitapi")
        click.echo(f"{info.project_name} {info.version}")


@cli.command()
@click.option("-g", "--glob", type=click.STRING, default="*", help="Globing pattern")
def activities(glob: str):
    click.echo(
        "\n".join(x for x in ApiOperator().activities() if fnmatch.fnmatch(x, glob))
    )


@cli.command()
@click.option(
    "-a",
    "--activity",
    type=click.STRING,
    required=True,
    help="Activity Uri (tenant/activity) (globbing)",
)
@click.option("-g", "--glob", type=click.STRING, default="*", help="Globing pattern")
def assignments(activity: str, glob: str):
    click.echo(
        "\n".join(
            x for x in ApiActivity(activity).assignments if fnmatch.fnmatch(x, glob)
        )
    )


@cli.command()
@click.option(
    "-a",
    "--activity",
    type=click.STRING,
    required=True,
    help="Activity Uri (tenant/activity) (globbing)",
)
@click.option("-g", "--glob", type=click.STRING, default="*", help="Globing pattern")
def submissions(activity: str, glob: str):
    click.echo(
        "\n".join(
            x for x in ApiActivity(activity).submissions_def if fnmatch.fnmatch(x, glob)
        )
    )


@cli.command()
@click.option(
    "-a",
    "--activity",
    type=click.STRING,
    required=True,
    help="Activity Uri (tenant/activity) (globbing)",
)
@click.option(
    "-l",
    "--logins",
    type=click.File("r"),
    default=None,
    help="List of logins (one per line)",
)
@click.option(
    "-f",
    "--filter",
    "filt",
    type=click.STRING,
    default="*",
    help="Globbing to filter submission definitions",
)
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
def grade(activity: str, logins: Optional[click.File], filt, select, mark, merge):
    logins_read = None
    res = {}
    if logins is not None:
        logins_read = Path(logins.name).read_text("utf-8").splitlines()
        res = {l: 0.0 for l in logins_read}
    a = ApiActivity(activity, logins_read)

    for k, v in grade_activity(a, filt, select, mark, merge).items():
        res[k] = v

    fname = a.activity.split("%2F")[1] + ".csv"
    print(f"Saving {len(res)} to {fname}")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# {' '.join(argv)}\n")
        f.writelines(f"{k},{v:.2f}\n" for k, v in res.items())


if __name__ == "__main__":
    cli()
