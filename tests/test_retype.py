#!/usr/bin/env python3

from functools import partial
from textwrap import dedent
from unittest import main, TestCase

from typed_ast import ast3

from retype import reapply, lib2to3_parse


class RetypeTestCase(TestCase):
    maxDiff = None

    def assertReapply(self, pyi_txt, src_txt, expected_txt):
        pyi = ast3.parse(dedent(pyi_txt))
        src = lib2to3_parse(dedent(src_txt))
        expected = lib2to3_parse(dedent(expected_txt))
        for node in pyi.body:
            reapply(node, src)
        self.longMessage = False
        self.assertEqual(expected, src, f"\n{expected!r} != \n{src!r}")


class ImportTestCase(RetypeTestCase):
    IMPORT = "import x"

    def _test_matched(self, matched, expected=None):
        pyi = f"{self.IMPORT}\n"
        src = f"{matched}\n"
        expected = f"{expected if expected is not None else matched}\n"
        self.assertReapply(pyi, src, expected)

    def _test_unmatched(self, unmatched):
        pyi = f"{self.IMPORT}\n"
        src = f"{unmatched}\n"
        expected = f"{unmatched}\n{self.IMPORT}\n"
        self.assertReapply(pyi, src, expected)

    def test_equal(self):
        self._test_matched(self.IMPORT)

    def test_src_empty(self):
        self._test_matched("", self.IMPORT)

    def test_matched1(self):
        self._test_matched("import x as x")

    def test_matched2(self):
        self._test_matched("import z, y, x")

    def test_matched3(self):
        self._test_matched("import z as y, x")

    def test_unmatched1(self):
        self._test_unmatched("import y as x")

    def test_unmatched2(self):
        self._test_unmatched("import x.y")

    def test_unmatched3(self):
        self._test_unmatched("import x.y as x")

    def test_unmatched4(self):
        self._test_unmatched("from x import x")

    def test_unmatched5(self):
        self._test_unmatched("from y import x")

    def test_unmatched6(self):
        self._test_unmatched("from . import x")

    def test_unmatched7(self):
        self._test_unmatched("from .x import x")


if __name__ == '__main__':
    main()
