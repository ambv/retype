from collections import namedtuple
from pathlib import Path

import pytest

from retype import walk_not_git_ignored


@pytest.fixture()  # type: ignore
def build(tmp_path):
    def _build(files):
        for file, content in files.items():
            dest = tmp_path / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        return tmp_path

    return _build


Case = namedtuple("Case", ["files", "found", "cwd"])
WALK_TESTS = {
    "no_ignore_root": Case({"a.py": ""}, ["a.py"], "."),
    "no_ignore_root_within": Case({"b/a.py": ""}, [str(Path("b") / "a.py")], "."),
    "no_ignore_keep_py_only": Case(
        {"a.py": "", "a.pyi": "", "b/a.pyi": "", "b/a.py": ""},
        ["a.py", str(Path("b") / "a.py")],
        ".",
    ),
    "ignore_py_at_root": Case(
        {".gitignore": "*.py", "a.py": "", "b/a.py": ""}, [], "."
    ),
    "ignore_py_nested": Case(
        {"a.py": "", "b/a.py": "", "b/.gitignore": "*.py"}, ["a.py"], "."
    ),
    "git_nested_no_res": Case(
        {".git/demo": "", ".gitignore": "*.py", "b/c/d/a.py": ""}, [], "b/c/d"
    ),
    "git_nested_has_res": Case(
        {".git/demo": "", "b/c/a.py": "", "b/c/d/.gitignore": "*.py", "b/c/d/a.py": ""},
        ["a.py"],
        "b/c",
    ),
}


@pytest.mark.parametrize(  # type: ignore
    "case", WALK_TESTS.values(), ids=list(WALK_TESTS.keys())
)
def test_walk(case: Case, build, monkeypatch):
    path = build(case.files)
    dest = path / case.cwd
    monkeypatch.chdir(dest)
    result = [
        str(f.relative_to(dest))
        for f in walk_not_git_ignored(
            dest, lambda p: p.suffix == ".py", extra_ignore=[]
        )
    ]
    expected = [str(f) for f in case.found]
    assert result == expected
