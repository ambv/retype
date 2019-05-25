import sys
from functools import partial
from pathlib import Path

import click

from . import retype_path
from .config import ReApplyFlags
from .version import __version__

Directory = partial(
    click.Path,
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    writable=False,
)


@click.command()
@click.option(
    "-p",
    "--pyi-dir",
    type=Directory(),
    default="types",
    help="Where to find .pyi stubs.",
    show_default=True,
)
@click.option(
    "-t",
    "--target-dir",
    type=Directory(exists=False, writable=True),
    default="typed-src",
    help="Where to write annotated sources.",
    show_default=True,
)
@click.option(
    "-i",
    "--incremental",
    is_flag=True,
    help="Allow for missing type annotations in both stubs and the source.",
)
@click.option("-q", "--quiet", is_flag=True, help="Don't emit warnings, just errors.")
@click.option(
    "-a", "--replace-any", is_flag=True, help="Allow replacing Any annotations."
)
@click.option(
    "--hg", is_flag=True, help="Post-process files to preserve implicit byte literals."
)
@click.option("--traceback", is_flag=True, help="Show a Python traceback on error.")
@click.argument("src", nargs=-1, type=Directory(file_okay=True))
@click.version_option(version=__version__)
def main(src, pyi_dir, target_dir, incremental, quiet, replace_any, hg, traceback):
    """Re-apply type annotations from .pyi stubs to your codebase."""

    exit_code = 0
    for src_entry in src:
        for file, error, exc_type, tb in retype_path(
            src=Path(src_entry),
            pyi_dir=Path(pyi_dir),
            targets=Path(target_dir),
            src_explicitly_given=True,
            quiet=quiet,
            hg=hg,
            flags=ReApplyFlags(replace_any=replace_any, incremental=incremental),
        ):
            print(f"error: {file}: {error}", file=sys.stderr)
            if traceback:
                print("Traceback (most recent call last):", file=sys.stderr)
                for line in tb:
                    print(line, file=sys.stderr, end="")
                print(f"{exc_type.__name__}: {error}", file=sys.stderr)
            exit_code += 1
    if not src and not quiet:
        print("warning: no sources given", file=sys.stderr)

    # According to http://tldp.org/LDP/abs/html/index.html starting with 126
    # we have special return codes.
    sys.exit(min(exit_code, 125))


if __name__ == "__main__":
    main()
