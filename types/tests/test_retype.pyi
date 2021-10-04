from lib2to3.pytree import Leaf, Node
from pathlib import Path
from typing import Tuple, Type, TypeVar, Union
from unittest import TestCase

from typed_ast import ast3

_E = TypeVar("_E", bound=Exception)
_LN = Union[Node, Leaf]

class RetypeTestCase(TestCase):
    def reapply(
        self,
        pyi_txt: str,
        src_txt: str,
        *,
        incremental: bool = ...,
        replace_any: bool = ...,
    ) -> Tuple[ast3.Module, Node]: ...
    def assertReapply(
        self,
        pyi_txt: str,
        src_txt: str,
        expected_txt: str,
        *,
        incremental: bool = ...,
        replace_any: bool = ...,
    ) -> None: ...
    def assertReapplyVisible(
        self,
        pyi_txt: str,
        src_txt: str,
        expected_txt: str,
        *,
        incremental: bool = ...,
        replace_any: bool = ...,
    ) -> None: ...
    def assertReapplyRaises(
        self,
        pyi_txt: str,
        src_txt: str,
        expected_exception: Type[_E],
        *,
        incremental: bool = ...,
        replace_any: bool = ...,
    ) -> _E: ...

def test_can_run_against_current_directory(tmp_path: Path) -> None: ...
def test_does_not_error_on_empty_file(tmp_path: Path) -> None: ...
