#!/usr/bin/env python3

from textwrap import dedent
from unittest import main, TestCase

from typed_ast import ast3

from retype import (
    Config,
    fix_remaining_type_comments,
    lib2to3_parse,
    reapply_all,
    serialize_attribute,
)


class RetypeTestCase(TestCase):
    maxDiff = None

    def assertReapply(self, pyi_txt, src_txt, expected_txt, *, incremental=False):
        Config.incremental = incremental
        pyi = ast3.parse(dedent(pyi_txt))
        src = lib2to3_parse(dedent(src_txt))
        expected = lib2to3_parse(dedent(expected_txt))
        assert isinstance(pyi, ast3.Module)
        reapply_all(pyi.body, src)
        fix_remaining_type_comments(src)
        self.longMessage = False
        self.assertEqual(expected, src, f"\n{expected!r} != \n{src!r}")

    def assertReapplyVisible(
        self, pyi_txt, src_txt, expected_txt, *, incremental=False
    ):
        Config.incremental = incremental
        pyi = ast3.parse(dedent(pyi_txt))
        src = lib2to3_parse(dedent(src_txt))
        expected = lib2to3_parse(dedent(expected_txt))
        assert isinstance(pyi, ast3.Module)
        reapply_all(pyi.body, src)
        fix_remaining_type_comments(src)
        self.longMessage = False
        self.assertEqual(
            str(expected),
            str(src),
            f"\n{str(expected)!r} != \n{str(src)!r}",
        )

    def assertReapplyRaises(
        self, pyi_txt, src_txt, expected_exception, *, incremental=False
    ):
        Config.incremental = incremental
        with self.assertRaises(expected_exception) as ctx:
            pyi = ast3.parse(dedent(pyi_txt))
            src = lib2to3_parse(dedent(src_txt))
            assert isinstance(pyi, ast3.Module)
            reapply_all(pyi.body, src)
            fix_remaining_type_comments(src)
        return ctx.exception


class ImportTestCase(RetypeTestCase):
    IMPORT = "import x"

    def _test_matched(self, matched: str, expected: str = None) -> None:
        pyi = f"{self.IMPORT}\n"
        src = f"{matched}\n"
        expected = f"{expected if expected is not None else matched}\n"
        self.assertReapply(pyi, src, expected)

    def _test_unmatched(self, unmatched: str) -> None:
        pyi = f"{self.IMPORT}\n"
        src = f"{unmatched}\n"
        expected = f"{unmatched}\n{self.IMPORT}\n"
        self.assertReapply(pyi, src, expected)

    def test_equal(self) -> None:
        self._test_matched(self.IMPORT)

    def test_src_empty(self) -> None:
        self._test_matched("", self.IMPORT)

    def test_matched1(self) -> None:
        self._test_matched("import x as x")

    def test_matched2(self) -> None:
        self._test_matched("import z, y, x")

    def test_matched3(self) -> None:
        self._test_matched("import z as y, x")

    def test_unmatched1(self) -> None:
        self._test_unmatched("import y as x")

    def test_unmatched2(self) -> None:
        self._test_unmatched("import x.y")

    def test_unmatched3(self) -> None:
        self._test_unmatched("import x.y as x")

    def test_unmatched4(self) -> None:
        self._test_unmatched("from x import x")

    def test_unmatched5(self) -> None:
        self._test_unmatched("from y import x")

    def test_unmatched6(self) -> None:
        self._test_unmatched("from . import x")

    def test_unmatched7(self) -> None:
        self._test_unmatched("from .x import x")


class FromImportTestCase(ImportTestCase):
    IMPORT = "from y import x"

    def test_matched1(self) -> None:
        self._test_matched("from y import x as x")

    def test_matched2(self) -> None:
        self._test_matched("from y import z, y, x")

    def test_matched3(self) -> None:
        self._test_matched("from y import z as y, x")

    def test_unmatched1(self) -> None:
        self._test_unmatched("from y import y as x")

    def test_unmatched2(self) -> None:
        self._test_unmatched("from y import x as y")

    def test_unmatched3(self) -> None:
        self._test_unmatched("from .y import x")

    def test_unmatched4(self) -> None:
        self._test_unmatched("import y.x")

    def test_unmatched5(self) -> None:
        self._test_unmatched("import y")

    def test_unmatched6(self) -> None:
        self._test_unmatched("from . import x")

    def test_unmatched7(self) -> None:
        self._test_unmatched("from .y import x")

    def test_unmatched8(self) -> None:
        self._test_unmatched("import x")


class FunctionReturnTestCase(RetypeTestCase):
    def test_missing_return_value_both(self) -> None:
        pyi_txt = "def fun(): ...\n"
        src_txt = "def fun(): ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: .pyi file is missing " +
            "return value and source doesn't provide it either",
            str(exception),
        )

    def test_missing_return_value_pyi(self) -> None:
        pyi_txt = "def fun(): ...\n"
        src_txt = "def fun() -> None: ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_missing_return_value_src(self) -> None:
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_return_value(self) -> None:
        pyi_txt = "def fun() -> List[Tuple[int, int]]: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> List[Tuple[int, int]]: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_return_value2(self) -> None:
        pyi_txt = "def fun() -> List[Tuple[Callable[[], Any], ...]]: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> List[Tuple[Callable[[], Any], ...]]: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_return_value3(self) -> None:
        pyi_txt = "def fun() -> List[Callable[[str, int, 'Custom'], Any]]: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> List[Callable[[str, int, 'Custom'], Any]]: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_mismatched_return_value(self) -> None:
        pyi_txt = "def fun() -> List[Tuple[int, int]]: ...\n"
        src_txt = "def fun() -> List[int]: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: incompatible existing " +
            "return value. Expected: 'List[Tuple[int, int]]', actual: 'List[int]'",
            str(exception),
        )

    def test_complex_return_value_type_comment(self) -> None:
        pyi_txt = """
        def fun():
            # type: () -> List[Callable[[str, int, 'Custom'], Any]]
            ...
        """
        src_txt = """
        def fun():
            ...
        """
        expected_txt = """
        def fun() -> List[Callable[[str, int, 'Custom'], Any]]:
            ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_return_value_spurious_type_comment(self) -> None:
        pyi_txt = """
        def fun():
            # type: () -> List[Callable[[str, int, 'Custom'], Any]]
            ...
        """
        src_txt = """
        def fun():
            # type: () -> List[Callable[[str, int, 'Custom'], Any]]
            ...
        """
        expected_txt = """
        def fun() -> List[Callable[[str, int, 'Custom'], Any]]:
            ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_missing_return_value_both_incremental(self) -> None:
        pyi_txt = "def fun(): ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun(): ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt, incremental=True)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt, incremental=True)

    def test_missing_return_value_pyi_incremental(self) -> None:
        pyi_txt = "def fun(): ...\n"
        src_txt = "def fun() -> None: ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt, incremental=True)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt, incremental=True)

    def test_missing_return_value_src_incremental(self) -> None:
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt, incremental=True)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt, incremental=True)


class FunctionArgumentTestCase(RetypeTestCase):
    def test_missing_ann_both(self) -> None:
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: .pyi file is missing " +
            "annotation for 'a1' and source doesn't provide it either",
            str(exception),
        )

    def test_missing_arg(self) -> None:
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a2) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: .pyi file expects " +
            "argument 'a1' next but argument 'a2' found in source",
            str(exception),
        )

    def test_missing_arg2(self) -> None:
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(*, a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "missing regular argument 'a1' in source",
            str(exception),
        )

    def test_missing_arg_kwonly(self) -> None:
        pyi_txt = "def fun(*, a1) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            ".pyi file expects *args or keyword-only arguments in source",
            str(exception),
        )

    def test_extra_arg1(self) -> None:
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "extra arguments in source: a1",
            str(exception),
        )

    def test_extra_arg2(self) -> None:
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(a1=None) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "extra arguments in source: a1=None",
            str(exception),
        )

    def test_extra_arg_kwonly(self) -> None:
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(*, a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "extra arguments in source: *, a1",
            str(exception),
        )

    def test_missing_default_arg_src(self) -> None:
        pyi_txt = "def fun(a1=None) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "source file does not specify default value for arg `a1` but the " +
            ".pyi file does",
            str(exception),
        )

    def test_missing_default_arg_pyi(self) -> None:
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1=None) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            ".pyi file does not specify default value for arg `a1` but the " +
            "source does",
            str(exception),
        )

    def test_missing_ann_pyi(self) -> None:
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1: str) -> None: ...\n"
        expected_txt = "def fun(a1: str) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_missing_ann_src(self) -> None:
        pyi_txt = "def fun(a1: str) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        expected_txt = "def fun(a1: str) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_args(self) -> None:
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun() -> None: ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_ann(self) -> None:
        pyi_txt = "def fun(a1: List[Tuple[int, int]]) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        expected_txt = "def fun(a1: List[Tuple[int, int]]) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_ann_with_default(self) -> None:
        pyi_txt = "def fun(a1: List[Tuple[int, int]] = None) -> None: ...\n"
        src_txt = "def fun(a1=None) -> None: ...\n"
        expected_txt = "def fun(a1: List[Tuple[int, int]] = None) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig1(self) -> None:
        pyi_txt = "def fun(a1: str, *args: str, kwonly1: int, **kwargs) -> None: ...\n"
        src_txt = "def fun(a1, *args, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "def fun(a1: str, *args: str, kwonly1: int = None, **kwargs) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig2(self) -> None:
        pyi_txt = "def fun(a1: str, *, kwonly1: int, **kwargs) -> None: ...\n"
        src_txt = "def fun(a1, *, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "def fun(a1: str, *, kwonly1: int = None, **kwargs) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig_async(self) -> None:
        pyi_txt = "async def fun(a1: str, *args: str, kwonly1: int, **kwargs) -> None: ...\n"
        src_txt = "async def fun(a1, *args, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "async def fun(a1: str, *args: str, kwonly1: int = None, **kwargs) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig1_type_comment(self) -> None:
        pyi_txt = """
        def fun(a1, *args, kwonly1, **kwargs):
            # type: (str, *str, int, **Any) -> None
            ...
        """
        src_txt = "def fun(a1, *args, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "def fun(a1: str, *args: str, kwonly1: int = None, **kwargs: Any) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig2_type_comment(self) -> None:
        pyi_txt = """
        def fun(a1, *, kwonly1, **kwargs):
            # type: (str, int, **Any) -> None
            ...
        """
        src_txt = "def fun(a1, *, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "def fun(a1: str, *, kwonly1: int = None, **kwargs: Any) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig3_type_comment(self) -> None:
        pyi_txt = """
        def fun(a1):
            # type: (Union[str, bytes]) -> None
            ...
        """
        src_txt = "def fun(a1): ...\n"
        expected_txt = "def fun(a1: Union[str, bytes]) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig4_type_comment(self) -> None:
        pyi_txt = """
        def fun(
            a1,  # type: str
            *,
            kwonly1,  # type: int
            **kwargs  # type: Any
        ):
            # type: (...) -> None
            ...
        """
        src_txt = "def fun(a1, *, kwonly1=None, **kwargs): ...\n"
        expected_txt = "def fun(a1: str, *, kwonly1: int = None, **kwargs: Any) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig4_spurious_type_comment(self) -> None:
        pyi_txt = """
        def fun(
            a1,  # type: str
            *,
            kwonly1,  # type: int
            **kwargs  # type: Any
        ):
            # type: (...) -> None
            ...
        """
        src_txt = """
        def fun(a1,
                *,
                kwonly1=None,  # type: int
                **kwargs
        ):
            ...
        """
        expected_txt = """
        def fun(a1: str,
                *,
                kwonly1: int = None,
                **kwargs: Any
        ) -> None:
            ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_missing_ann_both_incremental(self) -> None:
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        expected_txt= "def fun(a1) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt, incremental=True)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt, incremental=True)

    def test_missing_ann_both_multiple_args_incremental(self) -> None:
        pyi_txt = "def fun(a1, a2, *a3, **a4) -> None: ...\n"
        src_txt = "def fun(a1, a2, *a3, **a4) -> None: ...\n"
        expected_txt= "def fun(a1, a2, *a3, **a4) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt, incremental=True)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt, incremental=True)

    def test_missing_ann_both_incremental_default_value_whitespace(self) -> None:
        pyi_txt = "def fun(a1=..., a2: int = 0) -> None: ...\n"
        src_txt = "def fun(a1=False, a2=0) -> None: ...\n"
        expected_txt= "def fun(a1=False, a2: int = 0) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt, incremental=True)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt, incremental=True)

class FunctionVariableTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            name: str = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_default_type(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name: str
        """
        src_txt = """
        def fun():
            "Docstring"

            if False:
                name = "Dinsdale"
                print(name)
                name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"
            name: str
            if False:
                name = "Dinsdale"
                print(name)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_type_mismatch(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name: int = 0
            print(name)
            name = "Diinsdaalee"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 2:1: " +
            "incompatible existing variable annotation for 'name'. " +
            "Expected: 'str', actual: 'int'",
            str(exception),
        )

    def test_complex(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name: str
            age: int
            likes_spam: bool
        """
        src_txt = """
        def fun():
            "Docstring"

            name = "Dinsdale"
            print(name)
            if False:
                age = 100
                name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"
            age: int
            likes_spam: bool
            name: str = "Dinsdale"
            print(name)
            if False:
                age = 100
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_type(self) -> None:
        pyi_txt = """
        def fun() -> None:
            many_things: Union[List[int], str, 'Custom', Tuple[int, ...]]
        """
        src_txt = """
        def fun():
            "Docstring"

            many_things = []
            other_code()
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            many_things: Union[List[int], str, 'Custom', Tuple[int, ...]] = []
            other_code()
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class FunctionVariableTypeCommentTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name = ...  # type: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            name: str = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name = ...  # type: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value_type_comment(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name = ...  # type: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name = ...  # type: str
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_default_type(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name = ...  # type: str
        """
        src_txt = """
        def fun():
            "Docstring"

            if False:
                name = "Dinsdale"
                print(name)
                name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"
            name: str
            if False:
                name = "Dinsdale"
                print(name)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_type_mismatch(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name = ...  # type: str
        """
        src_txt = """
        def fun():
            "Docstring"

            name: int = 0
            print(name)
            name = "Diinsdaalee"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 2:1: " +
            "incompatible existing variable annotation for 'name'. " +
            "Expected: 'str', actual: 'int'",
            str(exception),
        )

    def test_complex(self) -> None:
        pyi_txt = """
        def fun() -> None:
            name = ...  # type: str
            age = ...  # type: int
            likes_spam = ...  # type: bool
        """
        src_txt = """
        def fun():
            "Docstring"

            name = "Dinsdale"
            print(name)
            if False:
                age = 100
                name = "Diinsdaalee"
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"
            age: int
            likes_spam: bool
            name: str = "Dinsdale"
            print(name)
            if False:
                age = 100
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_type(self) -> None:
        pyi_txt = """
        def fun() -> None:
            many_things = ...  # type: Union[List[int], str, 'Custom', Tuple[int, ...]]
        """
        src_txt = """
        def fun():
            "Docstring"

            many_things = []
            other_code()
        """
        expected_txt = """
        def fun() -> None:
            "Docstring"

            many_things: Union[List[int], str, 'Custom', Tuple[int, ...]] = []
            other_code()
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class MethodTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    super().__init__()
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    super().__init__()
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_two_classes(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None: ...
            class D:
                def __init__(self, a1: C, **kwargs) -> None: ...
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    super().__init__()

            class D:
                def __init__(self, a1, **kwargs) -> None:
                    super().__init__()
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    super().__init__()

            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    super().__init__()
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_function(self) -> None:
        pyi_txt = """
            class C:
                def method(self, a1: str, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            def method(self, a1, *args, kwonly1):
                print("I am not a method")

            class C:
                def method(self, a1, *args, kwonly1) -> None:
                    print("I am a method!")
        """
        expected_txt = """
            def method(self, a1, *args, kwonly1):
                print("I am not a method")

            class C:
                def method(self, a1: str, *args: str, kwonly1: int) -> None:
                    print("I am a method!")
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_missing_class(self) -> None:
        pyi_txt = """
            class C:
                def method(self, a1: str, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            def method(self, a1, *args, kwonly1):
                print("I am not a method")
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Class 'C' not found in source.",
            str(exception),
        )

    def test_staticmethod(self) -> None:
        pyi_txt = """
            class C:
                @yeah.what.aboutThis()
                @staticmethod
                def method(a1, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            class C:
                @whatAboutThis()
                @yeah
                @staticmethod
                def method(a1, *args, kwonly1) -> None:
                    print("I am a staticmethod, don't use me!")
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'method': 6:1: .pyi file is " +
            "missing annotation for 'a1' and source doesn't provide it either",
            str(exception),
        )

    def test_decorator_mismatch(self) -> None:
        pyi_txt = """
            class C:
                @yeah.what.aboutThis()
                @staticmethod
                def method(a1, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            class C:
                @classmethod
                def method(cls, a1, *args, kwonly1) -> None:
                    print("I am a staticmethod, don't use me!")
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Incompatible method kind for 'method': 4:1: Expected: " +
            "staticmethod, actual: classmethod",
            str(exception),
        )

    def test_decorator_mismatch2(self) -> None:
        pyi_txt = """
            class C:
                @staticmethod
                def method(a1, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            class C:
                @this.isnt.a.staticmethod
                def method(self, a1, *args, kwonly1) -> None:
                    print("I am a fake staticmethod, don't use me!")
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Incompatible method kind for 'method': 4:1: Expected: " +
            "staticmethod, actual: instancemethod",
            str(exception),
        )

    def test_decorator_mismatch3(self) -> None:
        pyi_txt = """
            class C:
                @this.isnt.a.staticmethod
                def method(a1, *args: str, kwonly1: int) -> None: ...
        """
        src_txt = """
            class C:
                @staticmethod
                def method(a1, *args, kwonly1) -> None:
                    print("I am a staticmethod, don't use me!")
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Incompatible method kind for 'method': 4:1: Expected: " +
            "instancemethod, actual: staticmethod",
            str(exception),
        )

    def test_complex_sig1_type_comment(self) -> None:
        pyi_txt = """
        class C:
            @staticmethod
            def fun(a1, *args, kwonly1, **kwargs):
                # type: (str, *str, int, **Any) -> None
                ...
        """
        src_txt = """
        class C:
            @staticmethod
            def fun(a1, *args, kwonly1=None, **kwargs):
                ...
        """
        expected_txt = """
        class C:
            @staticmethod
            def fun(a1: str, *args: str, kwonly1: int = None, **kwargs: Any) -> None:
                ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig2_type_comment(self) -> None:
        pyi_txt = """
        class C:
            def fun(self, a1, *, kwonly1, **kwargs):
                # type: (str, int, **Any) -> None
                ...
        """
        src_txt = """
        class C:
            def fun(self, a1, *, kwonly1=None, **kwargs):
                ...
        """
        expected_txt = """
        class C:
            def fun(self, a1: str, *, kwonly1: int = None, **kwargs: Any) -> None:
                ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig3_type_comment(self) -> None:
        pyi_txt = """
        class C:
            @staticmethod
            def fun(a1):
                # type: (Union[str, bytes]) -> None
                ...
        """
        src_txt = """
        class C:
            @staticmethod
            def fun(a1):
                ...
        """
        expected_txt = """
        class C:
            @staticmethod
            def fun(a1: Union[str, bytes]) -> None:
                ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig4_type_comment(self) -> None:
        pyi_txt = """
        class C:
            @classmethod
            def fun(cls, a1):
                # type: (Union[str, bytes]) -> None
                ...
        """
        src_txt = """
        class C:
            @classmethod
            def fun(cls, a1):
                ...
        """
        expected_txt = """
        class C:
            @classmethod
            def fun(cls, a1: Union[str, bytes]) -> None:
                ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig5_type_comment(self) -> None:
        pyi_txt = """
        class C:
            @classmethod
            def fun(cls, a1):
                # type: (Type['C'], Union[str, bytes]) -> None
                ...
        """
        src_txt = """
        class C:
            @classmethod
            def fun(cls, a1):
                ...
        """
        expected_txt = """
        class C:
            @classmethod
            def fun(cls: Type['C'], a1: Union[str, bytes]) -> None:
                ...
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class MethodVariableTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name: str
                age: int
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name = "Dinsdale"
                age = 47
                print(name, age)
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"

                name: str = "Dinsdale"
                age: int = 47
                print(name, age)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name: str
                print(name)
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"

                name: str
                print(name)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_default_type(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                if False:
                    name = "Dinsdale"
                    print(name)
                    name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"
                name: str
                if False:
                    name = "Dinsdale"
                    print(name)
                    name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_type_mismatch(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name: int = 0
                print(name)
                name = "Diinsdaalee"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 3:1: " +
            "incompatible existing variable annotation for 'name'. " +
            "Expected: 'str', actual: 'int'",
            str(exception),
        )

    def test_complex(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name: str
                age: int
                likes_spam: bool
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name = "Dinsdale"
                print(name)
                if False:
                    age = 100
                    name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"
                age: int
                likes_spam: bool
                name: str = "Dinsdale"
                print(name)
                if False:
                    age = 100
                    name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class MethodVariableTypeCommentTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name = ...  # type: str
                age = ...  # type: int
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name = "Dinsdale"
                age = 47
                print(name, age)
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"

                name: str = "Dinsdale"
                age: int = 47
                print(name, age)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name = ...  # type: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name: str
                print(name)
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"

                name: str
                print(name)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value_type_comment(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name = ...  # type: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name = ...  # type: str
                print(name)
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"

                name: str
                print(name)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_default_type(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name = ...  # type: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                if False:
                    name = "Dinsdale"
                    print(name)
                    name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"
                name: str
                if False:
                    name = "Dinsdale"
                    print(name)
                    name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_type_mismatch(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name = ...  # type: str
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name: int = 0
                print(name)
                name = "Diinsdaalee"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 3:1: " +
            "incompatible existing variable annotation for 'name'. " +
            "Expected: 'str', actual: 'int'",
            str(exception),
        )

    def test_complex(self) -> None:
        pyi_txt = """
        class C:
            def fun(self) -> None:
                name = ...  # type: str
                age = ...  # type: int
                likes_spam = ...  # type: bool
        """
        src_txt = """
        class C:
            def fun(self):
                "Docstring"

                name = "Dinsdale"
                print(name)
                if False:
                    age = 100
                    name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            def fun(self) -> None:
                "Docstring"
                age: int
                likes_spam: bool
                name: str = "Dinsdale"
                print(name)
                if False:
                    age = 100
                    name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class ModuleLevelVariableTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
        name: str
        """
        src_txt = """
        "Docstring"

        name = "Dinsdale"
        print(name)
        name = "Diinsdaalee"
        """
        expected_txt = """
        "Docstring"

        name: str = "Dinsdale"
        print(name)
        name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value(self) -> None:
        pyi_txt = """
        name: str
        """
        src_txt = """
        "Docstring"

        name: str
        print(name)
        name = "Diinsdaalee"
        """
        expected_txt = """
        "Docstring"

        name: str
        print(name)
        name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_default_type(self) -> None:
        pyi_txt = """
        name: str
        """
        src_txt = """
        "Docstring"

        if False:
            name = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        "Docstring"
        name: str
        if False:
            name = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_type_mismatch(self) -> None:
        pyi_txt = """
        name: str
        """
        src_txt = """
        "Docstring"

        name: int = 0
        print(name)
        name = "Diinsdaalee"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "incompatible existing variable annotation for 'name'. " +
            "Expected: 'str', actual: 'int'",
            str(exception),
        )

    def test_complex(self) -> None:
        pyi_txt = """
        name: str
        age: int
        likes_spam: bool
        """
        src_txt = """
        "Docstring"

        name = "Dinsdale"
        print(name)
        if False:
            age = 100
            name = "Diinsdaalee"
        """
        expected_txt = """
        "Docstring"
        age: int
        likes_spam: bool
        name: str = "Dinsdale"
        print(name)
        if False:
            age = 100
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_with_imports(self) -> None:
        pyi_txt = """
        from typing import Optional

        name: Optional[str]
        age: int
        likes_spam: bool
        """
        src_txt = """
        "Docstring"

        import sys

        name = "Dinsdale"
        print(name)
        if False:
            age = 100
            name = "Diinsdaalee"
        """
        expected_txt = """
        "Docstring"

        import sys

        from typing import Optional
        age: int
        likes_spam: bool
        name: Optional[str] = "Dinsdale"
        print(name)
        if False:
            age = 100
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_alias_basic(self) -> None:
        pyi_txt = """
        from typing import List, Optional

        MaybeStrings = Optional[List[Optional[str]]]
        SOME_GLOBAL: int

        def fun(errors: MaybeStrings) -> None: ...
        """
        src_txt = """
        "Docstring"

        from __future__ import print_function

        import sys

        SOME_GLOBAL: int = 0

        def fun(errors):
            for error in errors:
                if not error:
                    continue
                print(error, file=sys.stderr)
        """
        expected_txt = """
        "Docstring"

        from __future__ import print_function

        import sys

        from typing import List, Optional
        SOME_GLOBAL: int = 0
        MaybeStrings = Optional[List[Optional[str]]]

        def fun(errors: MaybeStrings) -> None:
            for error in errors:
                if not error:
                    continue
                print(error, file=sys.stderr)
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_alias_typevar(self) -> None:
        pyi_txt = """
        from typing import TypeVar

        _T = TypeVar('_T', bound=str)
        SOME_GLOBAL: int

        def fun(error: _T) -> _T: ...
        """
        src_txt = """
        "Docstring"

        from __future__ import print_function

        import sys

        SOME_GLOBAL: int = 0

        def fun(error):
            return error
        """
        expected_txt = """
        "Docstring"

        from __future__ import print_function

        import sys

        from typing import TypeVar
        SOME_GLOBAL: int = 0
        _T = TypeVar('_T', bound=str)

        def fun(error: _T) -> _T:
            return error
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_alias_typevar_typing(self) -> None:
        pyi_txt = """
        import typing.foo.bar

        _T = typing.foo.bar.TypeVar('_T', bound=str)
        SOME_GLOBAL: int

        def fun(error: _T) -> _T: ...
        """
        src_txt = """
        "Docstring"

        import sys

        SOME_GLOBAL: int = 0

        def fun(error):
            return error
        """
        expected_txt = """
        "Docstring"

        import sys

        import typing.foo.bar
        SOME_GLOBAL: int = 0
        _T = typing.foo.bar.TypeVar('_T', bound=str)

        def fun(error: _T) -> _T:
            return error
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_alias_many(self) -> None:
        pyi_txt = """
        from typing import TypeVar

        _T = TypeVar('_T', bound=str)
        _EitherStr = Union[str, bytes]
        _MaybeStrings = List[Optional[_EitherStr]]
        SOME_GLOBAL: int

        def fun(error: _T) -> _T: ...
        def fun2(errors: _MaybeStrings) -> None: ...
        """
        src_txt = """
        "Docstring"

        from __future__ import print_function

        import sys

        SOME_GLOBAL: int = 0

        def fun(error):
            return error

        @decorator
        def fun2(errors) -> None:
            for error in errors:
                if not error:
                    continue
                print(error, file=sys.stderr)
        """
        expected_txt = """
        "Docstring"

        from __future__ import print_function

        import sys

        from typing import TypeVar
        SOME_GLOBAL: int = 0
        _T = TypeVar('_T', bound=str)

        def fun(error: _T) -> _T:
            return error

        _EitherStr = Union[str, bytes]
        _MaybeStrings = List[Optional[_EitherStr]]
        @decorator
        def fun2(errors: _MaybeStrings) -> None:
            for error in errors:
                if not error:
                    continue
                print(error, file=sys.stderr)
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class ClassVariableTestCase(RetypeTestCase):
    def test_basic(self) -> None:
        pyi_txt = """
        class C:
            name: str
        """
        src_txt = """
        class C:
            "Docstring"

            name = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            "Docstring"

            name: str = "Dinsdale"
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_no_value(self) -> None:
        pyi_txt = """
        class C:
            name: str
        """
        src_txt = """
        class C:
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            "Docstring"

            name: str
            print(name)
            name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_default_type(self) -> None:
        pyi_txt = """
        class C:
            name: str
        """
        src_txt = """
        class C:
            "Docstring"

            if False:
                name = "Dinsdale"
                print(name)
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            "Docstring"
            name: str
            if False:
                name = "Dinsdale"
                print(name)
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_type_mismatch(self) -> None:
        pyi_txt = """
        class C:
            name: str
        """
        src_txt = """
        class C:
            "Docstring"

            name: int = 0
            print(name)
            name = "Diinsdaalee"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "incompatible existing variable annotation for 'name'. " +
            "Expected: 'str', actual: 'int'",
            str(exception),
        )

    def test_complex(self) -> None:
        pyi_txt = """
        class C:
            name: str
            age = ...  # type: int
            likes_spam: bool
        """
        src_txt = """
        class C:
            "Docstring"

            name = "Dinsdale"
            print(name)
            if False:
                age = 100
                name = "Diinsdaalee"
        """
        expected_txt = """
        class C:
            "Docstring"
            age: int
            likes_spam: bool
            name: str = "Dinsdale"
            print(name)
            if False:
                age = 100
                name = "Diinsdaalee"
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_instance_fields_no_assignment(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2 = ...  # type: int
                    self.field3: bool
            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    "Creates C."
                    super().__init__()

            class D:
                def __init__(self, a1, **kwargs) -> None:
                    "Creates D."
                    super().__init__()
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    "Creates C."
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
                    self.field3: bool
                    super().__init__()

            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    "Creates D."
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
                    super().__init__()
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_instance_fields_no_assignment_no_docstring(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2 = ...  # type: int
                    self.field3: bool
            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    super().__init__()

            class D:
                def __init__(self, a1, **kwargs) -> None:
                    super().__init__()
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
                    self.field3: bool
                    super().__init__()

            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
                    super().__init__()
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_instance_fields_no_assignment_docstring(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2 = ...  # type: int
                    self.field3: bool
            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    "Docstring"
                    super().__init__()

            class D:
                def __init__(self, a1, **kwargs) -> None:
                    "Docstring"
                    super().__init__()
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    "Docstring"
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
                    self.field3: bool
                    super().__init__()

            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    "Docstring"
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
                    super().__init__()
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_instance_fields_assignment_docstring(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2 = ...  # type: int
                    self.field3: bool
            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    "Docstring"
                    super().__init__()
                    self.field2 = 0
                    self.field1 = a1
                    print("unrelated instruction")
                    self.field0.subfield1.subfield2 = args[self.field2]

            class D:
                def __init__(self, a1, **kwargs) -> None:
                    "Docstring"
                    super().__init__()
                    self.field2 = None
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    "Docstring"
                    self.field3: bool
                    super().__init__()
                    self.field2: int = 0
                    self.field1: str = a1
                    print("unrelated instruction")
                    self.field0.subfield1.subfield2: Tuple[int] = args[self.field2]

            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    "Docstring"
                    self.field1: float
                    self.field3: int
                    super().__init__()
                    self.field2: Optional[str] = None
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_instance_fields_assignment_no_docstring(self) -> None:
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2 = ...  # type: int
                    self.field3: bool
            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field1: float
                    self.field2: Optional[str]
                    self.field3: int
        """
        src_txt = """
            class C:
                def __init__(self, a1, *args, kwonly1) -> None:
                    super().__init__()
                    self.field2 = 0
                    self.field1 = a1
                    print("unrelated instruction")
                    self.field0.subfield1.subfield2 = args[self.field2]

            class D:
                def __init__(self, a1, **kwargs) -> None:
                    super().__init__()
                    self.field2 = None
                    self.field1 = ...  # type: float
        """
        expected_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field3: bool
                    super().__init__()
                    self.field2: int = 0
                    self.field1: str = a1
                    print("unrelated instruction")
                    self.field0.subfield1.subfield2: Tuple[int] = args[self.field2]

            class D:
                def __init__(self, a1: C, **kwargs) -> None:
                    self.field3: int
                    super().__init__()
                    self.field2: Optional[str] = None
                    self.field1: float
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class SerializeTestCase(RetypeTestCase):
    def test_serialize_attribute(self) -> None:
        src_txt = "a.b.c"
        expected = "a.b.c"

        src = ast3.parse(dedent(src_txt))
        assert isinstance(src, ast3.Module)
        attr_expr = src.body[0]
        self.assertEqual(serialize_attribute(attr_expr), expected)

    def test_serialize_name(self) -> None:
        src_txt = "just_a_flat_name"
        expected = "just_a_flat_name"

        src = ast3.parse(dedent(src_txt))
        assert isinstance(src, ast3.Module)
        attr_expr = src.body[0]
        self.assertEqual(serialize_attribute(attr_expr), expected)


class PrintStmtTestCase(RetypeTestCase):
    def test_print_stmt_crash(self) -> None:
        pyi_txt = "def f() -> None: ...\n"
        src_txt = """
        import sys

        def f():
            print >>sys.stderr, "Nope"  # funnily, this parses just fine.
            print "This", "will", "fail", "to", "parse"
        """
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            'Cannot parse: 6:10:     print "This", "will", "fail", "to", "parse"',
            str(exception),
        )

class ParseErrorTestCase(RetypeTestCase):
    def test_missing_trailing_newline_crash(self) -> None:
        pyi_txt = "def f() -> None: ...\n"
        src_txt = """
        def f():
            pass"""
        expected_txt = """
        def f() -> None:
            pass
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

class PostProcessTestCase(RetypeTestCase):
    def test_straddling_variable_comments(self) -> None:
        pyi_txt = """
        def f(s: str) -> str: ...

        class C:
            def g(self) -> Iterator[Dict[int, str]]: ...
        """
        src_txt = """
        import sys

        def f(s):
            if s:
                l = []  # type: List[str]
                for elem in l:
                    s += elem
            return s

        class C:
            def g(self):
                for i in range(10):
                    result = {}  # type: Dict[int, str]
                    result[i] = f(str(i))
                    yield result
        """
        expected_txt = """
        import sys

        def f(s: str) -> str:
            if s:
                l: List[str] = []
                for elem in l:
                    s += elem
            return s

        class C:
            def g(self) -> Iterator[Dict[int, str]]:
                for i in range(10):
                    result: Dict[int, str] = {}
                    result[i] = f(str(i))
                    yield result
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_straddling_function_signature_type_comments1(self) -> None:
        pyi_txt = """
        class C:
            def f(self) -> Callable[[int, int], str]: ...
        """
        src_txt = """
        import sys

        class C:
            def f(self):
                def g(row, column):
                    # type: (int1, int2) -> str
                    return self.chessboard[row][column]
                return g
        """
        expected_txt = """
        import sys

        class C:
            def f(self) -> Callable[[int, int], str]:
                def g(row: int1, column: int2) -> str:
                    return self.chessboard[row][column]
                return g
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_straddling_function_signature_type_comments2(self) -> None:
        pyi_txt = """
        class C:
            def f(self) -> Callable[[int, int], str]: ...
        """
        src_txt = """
        import sys

        class C:
            def f(self):
                @some_decorator
                def g(row, column):
                    # type: (int1, int2) -> str
                    return self.chessboard[row][column]
                return g
        """
        expected_txt = """
        import sys

        class C:
            def f(self) -> Callable[[int, int], str]:
                @some_decorator
                def g(row: int1, column: int2) -> str:
                    return self.chessboard[row][column]
                return g
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_straddling_function_signature_type_comments3(self) -> None:
        pyi_txt = """
        class C:
            def f(self) -> Callable[[int, int], str]: ...
        """
        src_txt = """
        import sys

        class C:
            def f(self):
                def g(row, # type: int1
                      column, # type: int2
                ):
                    # type: (...) -> str
                    return self.chessboard[row][column]
                return g
        """
        expected_txt = """
        import sys

        class C:
            def f(self) -> Callable[[int, int], str]:
                def g(row: int1,
                      column: int2
                ) -> str:
                    return self.chessboard[row][column]
                return g
        """
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_straddling_function_signature_type_ignore(self) -> None:
        pyi_txt = """
        class C:
            def f(self) -> Callable[[int, int], str]: ...
        """
        src_txt = """
        import sys

        class C:
            def f(self):
                def g(row, # type: int1
                      column, # type: ignore
                ):
                    # type: ignore
                    return self.chessboard[row][column]
                return g
        """
        expected_txt = """
        import sys

        class C:
            def f(self) -> Callable[[int, int], str]:
                def g(row, # type: int1
                      column, # type: ignore
                ):
                    # type: ignore
                    return self.chessboard[row][column]
                return g
        """
        # NOTE: `# type: int1` is not applied either because of the missing
        # return value type comment.
        self.assertReapply(pyi_txt, src_txt, expected_txt)
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


if __name__ == '__main__':
    main()
