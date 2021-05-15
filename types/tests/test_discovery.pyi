from pathlib import Path
from typing import Callable, Dict, NamedTuple

from _pytest.monkeypatch import MonkeyPatch

Case: NamedTuple = ...

WALK_TESTS: Dict[str, Case]  # type: ignore # retype does not support merging class variant namedtuple with inline one

def build(tmp_path: Path) -> Callable[[Dict[str, str]], Path]:
    def _build(files: Dict[str, str]) -> Path: ...
    return _build

def test_walk(
    case: Case,  # type: ignore # retype does not support merging class variant namedtuple with inline one
    build: Callable[[Dict[str, str]], Path],
    monkeypatch: MonkeyPatch,
) -> None: ...
