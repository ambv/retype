from typing import Type, TypeVar
from unittest import TestCase

_E = TypeVar("_E", bound=Exception)

class RetypeTestCase(TestCase):
    def assertReapply(
        self, pyi_txt: str, src_txt: str, expected_txt: str, *, incremental: bool
    ) -> None: ...
    def assertReapplyVisible(
        self, pyi_txt: str, src_txt: str, expected_txt: str, *, incremental: bool
    ) -> None: ...
    def assertReapplyRaises(
        self,
        pyi_txt: str,
        src_txt: str,
        expected_exception: Type[_E],
        *,
        incremental: bool,
    ) -> _E: ...
