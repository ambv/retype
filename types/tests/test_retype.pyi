from lib2to3.pytree import Leaf, Node
from typing import Tuple, Type, TypeVar, Union
from unittest import TestCase

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
    ) -> Tuple[_LN, _LN]: ...
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
