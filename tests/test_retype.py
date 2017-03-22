#!/usr/bin/env python3

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

    def assertReapplyVisible(self, pyi_txt, src_txt, expected_txt):
        pyi = ast3.parse(dedent(pyi_txt))
        src = lib2to3_parse(dedent(src_txt))
        expected = lib2to3_parse(dedent(expected_txt))
        for node in pyi.body:
            reapply(node, src)
        self.longMessage = False
        self.assertEqual(
            str(expected),
            str(src),
            f"\n{str(expected)!r} != \n{str(src)!r}",
        )

    def assertReapplyRaises(self, pyi_txt, src_txt, expected_exception):
        with self.assertRaises(expected_exception) as ctx:
            pyi = ast3.parse(dedent(pyi_txt))
            src = lib2to3_parse(dedent(src_txt))
            for node in pyi.body:
                reapply(node, src)
        return ctx.exception


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


class FromImportTestCase(ImportTestCase):
    IMPORT = "from y import x"

    def test_matched1(self):
        self._test_matched("from y import x as x")

    def test_matched2(self):
        self._test_matched("from y import z, y, x")

    def test_matched3(self):
        self._test_matched("from y import z as y, x")

    def test_unmatched1(self):
        self._test_unmatched("from y import y as x")

    def test_unmatched2(self):
        self._test_unmatched("from y import x as y")

    def test_unmatched3(self):
        self._test_unmatched("from .y import x")

    def test_unmatched4(self):
        self._test_unmatched("import y.x")

    def test_unmatched5(self):
        self._test_unmatched("import y")

    def test_unmatched6(self):
        self._test_unmatched("from . import x")

    def test_unmatched7(self):
        self._test_unmatched("from .y import x")

    def test_unmatched8(self):
        self._test_unmatched("import x")


class FunctionReturnTestCase(RetypeTestCase):
    def test_missing_return_value_both(self):
        pyi_txt = "def fun(): ...\n"
        src_txt = "def fun(): ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: .pyi file is missing " +
            "return value and source doesn't provide it either",
            str(exception),
        )

    def test_missing_return_value_pyi(self):
        pyi_txt = "def fun(): ...\n"
        src_txt = "def fun() -> None: ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_missing_return_value_src(self):
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_complex_return_value(self):
        # Note: the current tuple formatting is unfortunate, this is how
        # astunparse deals with it currently.
        pyi_txt = "def fun() -> List[Tuple[int, int]]: ...\n"
        src_txt = "def fun(): ...\n"
        expected_txt = "def fun() -> List[Tuple[(int, int)]]: ...\n"
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_mismatched_return_value(self):
        pyi_txt = "def fun() -> List[Tuple[int, int]]: ...\n"
        src_txt = "def fun() -> List[int]: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: incompatible existing " +
            "return value. Expected: 'List[Tuple[(int, int)]]', actual: 'List[int]'",
            str(exception),
        )


class FunctionArgumentTestCase(RetypeTestCase):
    def test_missing_ann_both(self):
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: .pyi file is missing " +
            "annotation for 'a1' and source doesn't provide it either",
            str(exception),
        )

    def test_missing_arg(self):
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a2) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: .pyi file expects " +
            "argument 'a1' next but argument 'a2' found in source",
            str(exception),
        )

    def test_missing_arg2(self):
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(*, a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "missing regular argument 'a1' in source",
            str(exception),
        )

    def test_missing_arg_kwonly(self):
        pyi_txt = "def fun(*, a1) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            ".pyi file expects *args or keyword-only arguments in source",
            str(exception),
        )

    def test_extra_arg1(self):
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "extra arguments in source: a1",
            str(exception),
        )

    def test_extra_arg2(self):
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(a1=None) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "extra arguments in source: a1=None",
            str(exception),
        )

    def test_extra_arg_kwonly(self):
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun(*, a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "extra arguments in source: *, a1",
            str(exception),
        )

    def test_missing_default_arg_src(self):
        pyi_txt = "def fun(a1=None) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            "source file does not specify default value for arg `a1` but the " +
            ".pyi file does",
            str(exception),
        )

    def test_missing_default_arg_pyi(self):
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1=None) -> None: ...\n"
        exception = self.assertReapplyRaises(pyi_txt, src_txt, ValueError)
        self.assertEqual(
            "Annotation problem in function 'fun': 1:1: " +
            ".pyi file does not specify default value for arg `a1` but the " +
            "source does",
            str(exception),
        )

    def test_missing_ann_pyi(self):
        pyi_txt = "def fun(a1) -> None: ...\n"
        src_txt = "def fun(a1: str) -> None: ...\n"
        expected_txt = "def fun(a1: str) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_missing_ann_src(self):
        pyi_txt = "def fun(a1: str) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        expected_txt = "def fun(a1: str) -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_no_args(self):
        pyi_txt = "def fun() -> None: ...\n"
        src_txt = "def fun() -> None: ...\n"
        expected_txt = "def fun() -> None: ...\n"
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_complex_ann(self):
        # Note: the current tuple formatting is unfortunate, this is how
        # astunparse deals with it currently.
        pyi_txt = "def fun(a1: List[Tuple[int, int]]) -> None: ...\n"
        src_txt = "def fun(a1) -> None: ...\n"
        expected_txt = "def fun(a1: List[Tuple[(int, int)]]) -> None: ...\n"
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_ann_with_default(self):
        # Note: the current tuple formatting is unfortunate, this is how
        # astunparse deals with it currently.
        pyi_txt = "def fun(a1: List[Tuple[int, int]] = None) -> None: ...\n"
        src_txt = "def fun(a1=None) -> None: ...\n"
        expected_txt = "def fun(a1: List[Tuple[(int, int)]] = None) -> None: ...\n"
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)

    def test_complex_sig1(self):
        pyi_txt = "def fun(a1: str, *args: str, kwonly1: int, **kwargs) -> None: ...\n"
        src_txt = "def fun(a1, *args, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "def fun(a1: str, *args: str, kwonly1: int = None, **kwargs) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)

    def test_complex_sig2(self):
        pyi_txt = "def fun(a1: str, *, kwonly1: int, **kwargs) -> None: ...\n"
        src_txt = "def fun(a1, *, kwonly1=None, **kwargs) -> None: ...\n"
        expected_txt = "def fun(a1: str, *, kwonly1: int = None, **kwargs) -> None: ...\n"  # noqa
        self.assertReapply(pyi_txt, src_txt, expected_txt)


class FunctionVariableTestCase(RetypeTestCase):
    def test_basic(self):
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

    def test_no_value(self):
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

    def test_default_type(self):
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

    def test_type_mismatch(self):
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

    def test_complex(self):
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


class MethodTestCase(RetypeTestCase):
    def test_basic(self):
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

    def test_two_classes(self):
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

    def test_function(self):
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

    def test_missing_class(self):
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

    def test_staticmethod(self):
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

    def test_decorator_mismatch(self):
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


class MethodVariableTestCase(RetypeTestCase):
    def test_basic(self):
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

    def test_no_value(self):
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

    def test_default_type(self):
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

    def test_type_mismatch(self):
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

    def test_complex(self):
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


class ModuleLevelVariableTestCase(RetypeTestCase):
    def test_basic(self):
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

    def test_no_value(self):
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

    def test_default_type(self):
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

    def test_type_mismatch(self):
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

    def test_complex(self):
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

    def test_complex_with_imports(self):
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
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


class ClassVariableTestCase(RetypeTestCase):
    def test_basic(self):
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

    def test_no_value(self):
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

    def test_default_type(self):
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

    def test_type_mismatch(self):
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

    def test_complex(self):
        pyi_txt = """
        class C:
            name: str
            age: int
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

    def test_instance_fields_no_assignment(self):
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
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

    def test_instance_fields_no_assignment_no_docstring(self):
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
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

    def test_instance_fields_no_assignment_docstring(self):
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
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

    def test_instance_fields_assignment_docstring(self):
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
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

    def test_instance_fields_assignment_no_docstring(self):
        pyi_txt = """
            class C:
                def __init__(self, a1: str, *args: str, kwonly1: int) -> None:
                    self.field0.subfield1.subfield2: Tuple[int]
                    self.field1: str
                    self.field2: int
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
                    self.field1: float
                    self.field3: int
                    super().__init__()
                    self.field2: Optional[str] = None
        """
        self.assertReapplyVisible(pyi_txt, src_txt, expected_txt)


if __name__ == '__main__':
    main()
