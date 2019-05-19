from pathlib import Path
from typing import Callable, Dict, NamedTuple

from _pytest.monkeypatch import MonkeyPatch

Case: NamedTuple = ...

WALK_TESTS: Dict[str, Case]

def build(tmp_path: Path) -> Callable[[Dict[str, str]], Path]:
    def _build(files: Dict[str, str]) -> Path: ...

def test_walk(
    case: Case, build: Callable[[Dict[str, str]], Path], monkeypatch: MonkeyPatch
) -> None: ...
