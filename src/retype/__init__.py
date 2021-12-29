#!/usr/bin/env python3
"""Re-apply type annotations from .pyi stubs to your codebase."""
from __future__ import annotations

import os
import re
import sys
import tokenize
import traceback
from functools import singledispatch
from lib2to3 import pygram, pytree
from lib2to3.pgen2 import driver, token
from lib2to3.pgen2.parse import ParseError
from lib2to3.pygram import python_symbols as syms
from lib2to3.pytree import Leaf, Node
from pathlib import Path

from pathspec import PathSpec
from typed_ast import ast3

from .config import ReApplyFlags
from .version import __version__


def retype_path(
    src,
    pyi_dir,
    targets,
    *,
    src_explicitly_given=False,
    quiet=False,
    hg=False,
    flags=None,
):
    """Recursively retype files or directories given. Generate errors."""
    src = src.absolute()
    if src.is_dir():
        extra_ignore = []
        for folder in [pyi_dir, targets]:
            try:
                extra_ignore.append(f"/{folder.relative_to(src)}")
            except ValueError:
                pass
        for file in walk_not_git_ignored(
            src, lambda p: p.suffix == ".py", extra_ignore
        ):
            nested = file.relative_to(src).parent
            yield from retype_path(
                file,
                pyi_dir / nested,
                targets / nested,
                quiet=quiet,
                hg=hg,
                flags=flags,
            )
    elif src.suffix == ".py" or src_explicitly_given:
        try:
            retype_file(src, pyi_dir, targets, quiet=quiet, hg=hg, flags=flags)
        except Exception as e:
            yield (src, str(e), type(e), traceback.format_tb(e.__traceback__))


def retype_file(src, pyi_dir, targets, *, quiet=False, hg=False, flags=None):
    """Retype `src`, finding types in `pyi_dir`. Save in `targets`.

    The file should remain formatted exactly as it was before, save for:
    - annotations
    - additional imports needed to satisfy annotations
    - additional module-level names needed to satisfy annotations

    Type comments in sources are normalized to type annotations.
    """
    if flags is None:
        flags = ReApplyFlags()
    with tokenize.open(src) as src_buffer:
        src_contents = src_buffer.read()
        if src_contents == "":
            return None
        src_encoding = src_buffer.encoding
        src_node = lib2to3_parse(src_contents)
    try:
        with open((pyi_dir / src.name).with_suffix(".pyi")) as pyi_file:
            pyi_txt = pyi_file.read()
    except FileNotFoundError:
        if not quiet:
            print(
                f"warning: .pyi file for source {src} not found in {pyi_dir}",
                file=sys.stderr,
            )
    else:
        pyi_ast = ast3.parse(pyi_txt)
        assert isinstance(pyi_ast, ast3.Module)
        reapply_all(pyi_ast.body, src_node, flags)
    fix_remaining_type_comments(src_node, flags)
    targets.mkdir(parents=True, exist_ok=True)
    with open(targets / src.name, "w", encoding=src_encoding) as target_file:
        target_file.write(lib2to3_unparse(src_node, hg=hg))
    return targets / src.name


def lib2to3_parse(src_txt):
    """Given a string with source, return the lib2to3 Node."""
    grammar = pygram.python_grammar_no_print_statement
    drv = driver.Driver(grammar, pytree.convert)
    if src_txt[-1] != "\n":
        nl = "\r\n" if "\r\n" in src_txt[:1024] else "\n"
        src_txt += nl
    try:
        result = drv.parse_string(src_txt, True)
    except ParseError as pe:
        lineno, column = pe.context[1]
        lines = src_txt.splitlines()
        try:
            faulty_line = lines[lineno - 1]
        except IndexError:
            faulty_line = "<line number missing in source>"
        raise ValueError(f"Cannot parse: {lineno}:{column}: {faulty_line}") from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])

    return result


def lib2to3_unparse(node, *, hg=False):
    """Given a lib2to3 node, return its string representation."""
    code = str(node)
    if hg:
        from retype.retype_hgext import apply_job_security

        code = apply_job_security(code)
    return code


def reapply_all(ast_node, lib2to3_node, flags):
    """Reapplies the typed_ast node into the lib2to3 tree.

    Also does post-processing. This is done in reverse order to enable placing
    TypeVars and aliases that depend on one another.
    """
    late_processing = reapply(ast_node, lib2to3_node, flags)
    for lazy_func in reversed(late_processing):
        lazy_func()


@singledispatch
def reapply(ast_node, lib2to3_node, flags):
    """Reapplies the typed_ast node into the lib2to3 tree.

    By default does nothing.
    """
    return []


@reapply.register(list)
def _r_list(l, lib2to3_node, flags):
    if lib2to3_node.type not in (syms.file_input, syms.suite):
        return []

    result = []
    for pyi_node in l:
        result.extend(reapply(pyi_node, lib2to3_node, flags))
    return result


@reapply.register(ast3.ImportFrom)
def _r_importfrom(import_from, node, flags):
    assert node.type in (syms.file_input, syms.suite)
    level = import_from.level or 0
    module = "." * level + (import_from.module or "")
    names = import_from.names
    for child in flatten_some(node.children):
        if child.type != syms.simple_stmt:
            continue

        stmt = child.children[0]
        if stmt.type == syms.import_from:
            imp = stmt.children
            # if the module we're looking for is already imported, skip it.
            if str(imp[1]).strip() == module and names_already_imported(names, imp[3]):
                break
    else:
        import_stmt = make_import(*names, from_module=module)
        append_after_imports(import_stmt, node)
    return []


@reapply.register(ast3.Import)
def _r_import(import_, node, flags):
    assert node.type in (syms.file_input, syms.suite)
    names = import_.names
    for child in flatten_some(node.children):
        if child.type != syms.simple_stmt:
            continue

        stmt = child.children[0]
        if stmt.type == syms.import_name:
            imp = stmt.children
            # if the module we're looking for is already imported, skip it.
            if names_already_imported(names, imp[1]):
                break
    else:
        import_stmt = make_import(*names)
        append_after_imports(import_stmt, node)
    return []


@reapply.register(ast3.ClassDef)
def _r_classdef(cls, node, flags):
    assert node.type in (syms.file_input, syms.suite)
    name = Leaf(token.NAME, cls.name)
    for child in flatten_some(node.children):
        if child.type == syms.decorated:
            # skip decorators
            child = child.children[1]
        if child.type == syms.classdef and child.children[1] == name:
            cls_node = child.children[-1]
            break
    else:
        raise ValueError(f"Class {name.value!r} not found in source.")

    result = []
    for ast_elem in cls.body:
        result.extend(reapply(ast_elem, cls_node, flags))
    return result


@reapply.register(ast3.AsyncFunctionDef)
@reapply.register(ast3.FunctionDef)
def _r_functiondef(fun, node, flags):
    assert node.type in (syms.file_input, syms.suite)
    name = Leaf(token.NAME, fun.name)
    pyi_decorators = decorator_names(fun.decorator_list)
    pyi_method_decorators = list(
        filter(is_builtin_method_decorator, pyi_decorators)
    ) or ["instancemethod"]
    is_method = (
        node.parent is not None
        and node.parent.type == syms.classdef
        and "staticmethod" not in pyi_method_decorators
    )
    args, returns = get_function_signature(fun, is_method=is_method)
    for child in flatten_some(node.children):
        decorators = None
        if child.type == syms.decorated:
            # skip decorators
            decorators = child.children[0]
            child = child.children[1]

        if child.type in (syms.async_stmt, syms.async_funcdef):
            # async def in 3.5 and 3.6
            child = child.children[1]

        if child.type != syms.funcdef:
            continue

        offset = 1
        if child.children[offset] == name:
            lineno = child.get_lineno()
            column = 1

            if decorators:
                src_decorators = decorator_names(decorators)
                src_method_decorators = list(
                    filter(is_builtin_method_decorator, src_decorators)
                ) or ["instancemethod"]
                if pyi_method_decorators != src_method_decorators:
                    raise ValueError(
                        f"Incompatible method kind for {fun.name!r}: "
                        + f"{lineno}:{column}: Expected: "
                        + f"{pyi_method_decorators[0]}, actual: "
                        + f"{src_method_decorators[0]}"
                    )

                is_method = "staticmethod" not in pyi_decorators

            try:
                annotate_parameters(
                    child.children[offset + 1], args, is_method=is_method, flags=flags
                )
                annotate_return(child.children, returns, offset + 2, flags)
                reapply(fun.body, child.children[-1], flags)
                remove_function_signature_type_comment(child.children[-1])
            except ValueError as ve:
                raise ValueError(
                    f"Annotation problem in function {name.value!r}: "
                    + f"{lineno}:{column}: {ve}"
                ) from ve
            break
    else:
        raise ValueError(f"Function {name.value!r} not found in source.")

    return []


@reapply.register(ast3.AnnAssign)
def _r_annassign(annassign, body, flags):
    assert body.type in (syms.file_input, syms.suite)

    target = annassign.target
    if isinstance(target, ast3.Name):
        name = target.id
    elif isinstance(target, ast3.Attribute):
        name = serialize_attribute(target)
    else:
        raise NotImplementedError(f"unexpected assignment target: {target}")

    annotation = convert_annotation(annassign.annotation)
    annotation.prefix = " "
    annassign_node = Node(syms.annassign, [new(_colon), annotation])
    for child in flatten_some(body.children):
        if child.type != syms.simple_stmt:
            continue

        maybe_expr = child.children[0]
        if maybe_expr.type != syms.expr_stmt:
            continue

        expr = maybe_expr.children

        if (
            expr[0].type in (token.NAME, syms.power)
            and minimize_whitespace(str(expr[0])) == name
        ):
            if expr[1].type == syms.annassign:
                # variable already typed, let's just ensure it's sane
                if len(expr[1].children) > 2 and expr[1].children[2] != _eq:
                    raise NotImplementedError(
                        f"unexpected element after annotation: {str(expr[3])}"
                    )
                expr[1].children[1] = maybe_replace_any_if_equal(
                    f"variable annotation for {name!r}",
                    annotation,
                    expr[1].children[1],
                    flags,
                )
                break

            if expr[1] != _eq:
                # If it's not an assignment, we're ignoring it. It could be:
                # - indexing
                # - tuple unpacking
                # - calls
                # - etc. etc.
                continue

            maybe_type_comment = _type_comment_re.match(child.children[1].prefix)
            if maybe_type_comment:
                # variable already typed by type comment, let's ensure it's sane...
                type_comment = parse_type_comment(maybe_type_comment.group("type"))
                actual_annotation = convert_annotation(type_comment)
                ensure_annotations_equal(
                    f"variable annotation for {name!r}",
                    annotation,
                    actual_annotation,
                    flags,
                )
                # ...and remove the redundant comment
                child.children[1].prefix = maybe_space_before_comment(
                    maybe_type_comment.group("nl")
                )

            if len(expr[2:]) > 0 and expr[2:] != [_ellipsis]:
                # copy the value unless it was an old-style variable type comment
                # with no actual value (but just a ... placeholder)
                annassign_node.children.append(new(_eq))
                annassign_node.children.extend(new(elem) for elem in expr[2:])

            maybe_expr.children = [expr[0], annassign_node]

            break
    else:
        # If the variable was used in some `if` statement, etc.; let's define
        # its type from the stub on the top level of the function.
        offset, prefix = get_offset_and_prefix(body, skip_assignments=True)
        body.children.insert(
            offset,
            Node(
                syms.simple_stmt,
                [
                    Node(syms.expr_stmt, [Leaf(token.NAME, name), annassign_node]),
                    new(_newline),
                ],
                prefix=prefix.lstrip("\n"),
            ),
        )

    return []


@reapply.register(ast3.Assign)
def _r_assign(assign, body, flags):
    assert body.type in (syms.file_input, syms.suite)

    if len(assign.targets) != 1:
        # Type aliases and old-style var type comments cannot have multiple
        # targets.
        return []

    if assign.type_comment:
        # old-style variable type comment, let's treat it exactly like
        # a new-style annotated assignment
        tc = parse_type_comment(assign.type_comment)
        annassign = ast3.AnnAssign(
            target=assign.targets[0], annotation=tc, value=assign.value, simple=False
        )
        return reapply(annassign, body, flags)

    if not isinstance(assign.targets[0], ast3.Name):
        # Type aliases cannot be attributes, etc.
        return []

    name = assign.targets[0].id
    value = convert_annotation(assign.value)
    value.prefix = " "

    for child in flatten_some(body.children):
        if child.type != syms.simple_stmt:
            continue

        maybe_expr = child.children[0]
        if maybe_expr.type != syms.expr_stmt:
            continue

        expr = maybe_expr.children

        if (
            isinstance(expr[0], Leaf)
            and expr[0].type == token.NAME
            and expr[0].value == name
            and expr[1] == _eq
        ):
            expr[2] = maybe_replace_any_if_equal(
                f"alias {name!r}", value, expr[2], flags=flags
            )
            break
    else:
        # We need to defer placing aliases because we need to place them
        # relative to their usage, and the type annotations likely come after
        # in the .pyi file.

        def lazy_aliasing() -> None:
            # We should find the first place where the alias is used and put it
            # right above.  This way we don't need to look at the value at all.
            _, prefix = get_offset_and_prefix(body, skip_assignments=True)
            name_node = Leaf(token.NAME, name)
            for _offset, stmt in enumerate(body.children):
                if name_used_in_node(stmt, name_node):
                    break
            else:
                _offset = -1

            body.children.insert(
                _offset,
                Node(
                    syms.simple_stmt,
                    [
                        Node(syms.expr_stmt, [Leaf(token.NAME, name), new(_eq), value]),
                        new(_newline),
                    ],
                    prefix=prefix.lstrip("\n"),
                ),
            )

        return [lazy_aliasing]

    return []


@singledispatch
def serialize_attribute(attr):
    """serialize_attribute(Attribute()) -> "self.f1.f2.f3"

    Change an AST object into its string representation."""
    return ""


@serialize_attribute.register(ast3.Attribute)
def _sa_attribute(attr):
    return f"{serialize_attribute(attr.value)}.{attr.attr}"


@serialize_attribute.register(ast3.Name)
def _sa_name(name):
    return name.id


@serialize_attribute.register(ast3.Expr)
def _sa_expr(expr):
    return serialize_attribute(expr.value)


def convert_annotation(ann):
    return normalize_node(_convert_annotation(ann))


def normalize_node(node):
    """Normalizes nodes to match pytree.convert and flatten power nodes"""

    if isinstance(node, Node):
        if len(node.children) == 1:
            return node.children[0]

        node.children = [normalize_node(i) for i in node.children]

    # if the node is a power node inline child power nodes
    if node.type == syms.power:
        children = []
        for child in node.children:
            if child.type == syms.power:
                children.extend(child.children)
            else:
                children.append(child)

        node.children = children

    return node


@singledispatch
def _convert_annotation(ann):
    """Converts an AST object into its lib2to3 equivalent."""
    raise NotImplementedError(f"unknown AST node type: {ann!r}")


@_convert_annotation.register(ast3.Subscript)
def _c_subscript(sub):
    return Node(
        syms.power,
        [
            _convert_annotation(sub.value),
            Node(
                syms.trailer, [new(_lsqb), _convert_annotation(sub.slice), new(_rsqb)]
            ),
        ],
    )


@_convert_annotation.register(ast3.Name)
def _c_name(name):
    return Leaf(token.NAME, name.id)


@_convert_annotation.register(ast3.NameConstant)
def _c_nameconstant(const):
    return Leaf(token.NAME, repr(const.value))


@_convert_annotation.register(ast3.Ellipsis)
def _c_ellipsis(ell):
    return Node(syms.atom, [new(_dot), new(_dot), new(_dot)])


@_convert_annotation.register(ast3.Str)
def _c_str(s):
    return Leaf(token.STRING, repr(s.s))


@_convert_annotation.register(ast3.Num)
def _c_num(n):
    return Leaf(token.NUMBER, repr(n.n))


@_convert_annotation.register(ast3.Index)
def _c_index(index):
    return _convert_annotation(index.value)


@_convert_annotation.register(ast3.Tuple)
def _c_tuple(tup):
    contents = [_convert_annotation(elt) for elt in tup.elts]
    for index in range(len(contents) - 1, 0, -1):
        contents[index].prefix = " "
        contents.insert(index, new(_comma))

    return Node(syms.subscriptlist, contents)


@_convert_annotation.register(ast3.Attribute)
def _c_attribute(attr):
    return Node(
        syms.power,
        [
            _convert_annotation(attr.value),
            Node(syms.trailer, [new(_dot), Leaf(token.NAME, attr.attr)]),
        ],
    )


@_convert_annotation.register(ast3.Call)
def _c_call(call):
    contents = [_convert_annotation(arg) for arg in call.args]
    contents.extend(_convert_annotation(kwarg) for kwarg in call.keywords)
    for index in range(len(contents) - 1, 0, -1):
        contents[index].prefix = " "
        contents.insert(index, new(_comma))

    call_args = [new(_lpar), new(_rpar)]
    if contents:
        call_args.insert(1, Node(syms.arglist, contents))
    return Node(
        syms.power, [_convert_annotation(call.func), Node(syms.trailer, call_args)]
    )


@_convert_annotation.register(ast3.keyword)
def _c_keyword(kwarg):
    assert kwarg.arg
    return Node(
        syms.argument,
        [
            Leaf(token.NAME, kwarg.arg),
            new(_eq, prefix=""),
            _convert_annotation(kwarg.value),
        ],
    )


@_convert_annotation.register(ast3.List)
def _c_list(l):
    contents = [_convert_annotation(elt) for elt in l.elts]
    for index in range(len(contents) - 1, 0, -1):
        contents[index].prefix = " "
        contents.insert(index, new(_comma))

    list_literal = [new(_lsqb), new(_rsqb)]
    if contents:
        list_literal.insert(1, Node(syms.listmaker, contents))
    return Node(syms.atom, list_literal)


@singledispatch
def names_already_imported(names, node):
    """Returns True if `node` represents `names`."""
    return False


@names_already_imported.register(list)
def _nai_list(names, node):
    return all(names_already_imported(name, node) for name in names)


@names_already_imported.register(ast3.alias)
def _nai_alias(alias, node):
    # Comments below show example imports that match the rule.
    name = Leaf(token.NAME, alias.name)
    if not alias.asname or alias.asname == alias.name:
        # import hay, x, stack
        # from field import hay, s, stack
        if node.type in (syms.dotted_as_names, syms.import_as_names):
            return name in node.children

        # import x as x
        # from field import x as x
        if node.type in (syms.dotted_as_name, syms.import_as_name):
            return [name, _as, name] == node.children

        # import x
        return node == name

    asname = Leaf(token.NAME, alias.asname)
    dotted_as_name = Node(syms.dotted_as_name, [name, _as, asname])
    # import hay as stack, x as y
    if node.type == syms.dotted_as_names:
        return dotted_as_name in node.children

    import_as_name = Node(syms.import_as_name, [name, _as, asname])
    # from field import hay as stack, x as y
    if node.type == syms.import_as_names:
        return import_as_name in node.children

    # import x as y
    # from field import x as y
    return node in (dotted_as_name, import_as_name)


@singledispatch
def decorator_names(obj):
    return []


@decorator_names.register(Node)
def _dn_node(node):
    if node.type == syms.decorator:
        return [str(node.children[1])]

    if node.type == syms.decorators:
        return [str(decorator.children[1]) for decorator in node.children]

    return []


@decorator_names.register(list)
def _dn_list(l):
    result = []
    for elem in l:
        result.extend(decorator_names(elem))
    return result


@decorator_names.register(ast3.Name)
def _dn_name(name):
    return [name.id]


@decorator_names.register(ast3.Call)
def _dn_call(call):
    return decorator_names(call.func)


@decorator_names.register(ast3.Attribute)
def _dn_attribute(attr):
    return [serialize_attribute(attr)]


def fix_remaining_type_comments(node, flags):
    """Converts type comments in `node` to proper annotated assignments."""
    assert node.type == syms.file_input

    last_n = None
    for n in node.post_order():
        if last_n is not None:
            if n.type == token.NEWLINE and is_assignment(last_n):
                fix_variable_annotation_type_comment(n, last_n)
            elif n.type == syms.funcdef and last_n.type == syms.suite:
                fix_signature_annotation_type_comment(n, last_n, offset=1, flags=flags)
            elif n.type == syms.async_funcdef and last_n.type == syms.suite:
                fix_signature_annotation_type_comment(n, last_n, offset=2, flags=flags)
        last_n = n


def fix_variable_annotation_type_comment(node, last):
    m = _type_comment_re.match(node.prefix)
    if not m:
        return

    type_comment = parse_type_comment(m.group("type"))
    ann = convert_annotation(type_comment)
    ann.prefix = " "
    annassign_node = Node(syms.annassign, [new(_colon), ann])
    expr = last.children
    if len(expr[2:]) > 0 and expr[2:] != [_ellipsis]:
        # with assignment
        annassign_node.children.append(new(_eq))
        annassign_node.children.extend(new(elem) for elem in expr[2:])
    last.children = [expr[0], annassign_node]
    node.prefix = maybe_space_before_comment(m.group("nl"))


def fix_signature_annotation_type_comment(node, last, *, offset, flags):
    for ch in last.children:
        if ch.type == token.INDENT:
            break
    else:
        return

    m = _type_comment_re.match(ch.prefix)
    if not m:
        return

    parameters = node.children[offset + 1]
    args_tc, returns_tc = parse_signature_type_comment(m.group("type"))
    ast_args = parse_arguments(str(parameters))
    # `is_method=True` below only means we allow for missing first annotation.
    # It's not even worth checking at this point.
    copy_arguments_to_annotations(ast_args, args_tc, is_method=True)
    annotate_parameters(parameters, ast_args, is_method=True, flags=flags)
    annotate_return(node.children, returns_tc, offset + 2, flags)
    remove_function_signature_type_comment(last)


def is_assignment(node):
    if node.type != syms.expr_stmt:
        return False

    expr = node.children
    # The `bool()` below shuts up a "returning Any" warning from mypy.
    return expr[0].type in (token.NAME, syms.power) and bool(expr[1] == _eq)


def is_builtin_method_decorator(name):
    return name in {"classmethod", "staticmethod"}


def make_import(*names, from_module=None):
    assert names
    imports = []

    if from_module:
        statement = syms.import_from
        container = syms.import_as_names
        single = syms.import_as_name
        result = [
            Leaf(token.NAME, "from"),
            Leaf(token.NAME, from_module, prefix=" "),
            Leaf(token.NAME, "import", prefix=" "),
        ]
    else:
        statement = syms.import_name
        container = syms.dotted_as_names
        single = syms.dotted_as_name
        result = [Leaf(token.NAME, "import")]

    for alias in names:
        name = Leaf(token.NAME, alias.name, prefix=" ")
        if alias.asname:
            _as = Leaf(token.NAME, "as", prefix=" ")
            asname = Leaf(token.NAME, alias.asname, prefix=" ")
            imports.append(Node(single, [name, _as, asname]))
        else:
            imports.append(name)
    if len(imports) == 1:
        result.append(imports[0])
    else:
        imports_and_commas = []
        for imp in imports[:-1]:
            imports_and_commas.append(imp)
            imports_and_commas.append(Leaf(token.COMMA, ","))
        imports_and_commas.append(imports[-1])
        result.append(Node(container, imports_and_commas))
    return Node(
        syms.simple_stmt,
        [Node(statement, result), Leaf(token.NEWLINE, "\n")],  # FIXME: \r\n?
    )


def append_after_imports(stmt_to_insert, node):
    offset, stmt_to_insert.prefix = get_offset_and_prefix(node)
    node.children.insert(offset, stmt_to_insert)


def annotate_parameters(parameters, ast_args, *, is_method=False, flags):
    params = parameters.children[1:-1]
    if len(params) == 0:
        return  # FIXME: handle checking if the expected (AST) function is also empty.
    elif len(params) > 1:
        raise NotImplementedError(f"unknown AST structure in parameters: {params}")

    # Simplify the possible data structures so we can just pull from it.
    if params[0].type == syms.typedargslist:
        params = params[0].children

    typedargslist = []

    num_args_no_defaults = len(ast_args.args) - len(ast_args.defaults)
    defaults = [None] * num_args_no_defaults
    defaults.extend(ast_args.defaults)
    typedargslist.extend(
        gen_annotated_params(
            ast_args.args, defaults, params, is_method=is_method, flags=flags
        )
    )
    hopefully_vararg = None
    if ast_args.vararg or ast_args.kwonlyargs:
        try:
            hopefully_star, hopefully_vararg = pop_param(params)
            if hopefully_star != _star:
                raise ValueError
        except (IndexError, ValueError):
            raise ValueError(
                ".pyi file expects *args or keyword-only arguments in source"
            ) from None
        else:
            typedargslist.append(new(_comma))
            typedargslist.append(new(hopefully_star))

    if ast_args.vararg and hopefully_vararg:
        if hopefully_vararg.type == syms.tname:
            assert isinstance(hopefully_vararg.children[0], Leaf)
            hopefully_vararg_name = hopefully_vararg.children[0].value
        else:
            assert isinstance(hopefully_vararg, Leaf)
            hopefully_vararg_name = hopefully_vararg.value
        if hopefully_vararg_name != ast_args.vararg.arg:
            raise ValueError(f".pyi file expects *{ast_args.vararg.arg} in source")

        typedargslist.append(
            get_annotated_param(
                hopefully_vararg, ast_args.vararg, missing_ok=True, flags=flags
            )
        )

    if ast_args.kwonlyargs:
        if not ast_args.vararg:
            if hopefully_vararg != _comma:
                raise ValueError(
                    ".pyi file expects keyword-only arguments but "
                    + f"*{str(hopefully_vararg).strip()} found in source"
                )

        typedargslist.extend(
            gen_annotated_params(
                ast_args.kwonlyargs,
                ast_args.kw_defaults,
                params,
                implicit_default=True,
                flags=flags,
            )
        )

    if ast_args.kwarg:
        try:
            hopefully_dstar, hopefully_kwarg = pop_param(params)
            if not hopefully_kwarg:
                raise ValueError
            if hopefully_kwarg.type == syms.tname:
                assert isinstance(hopefully_kwarg.children[0], Leaf)
                hopefully_kwarg_name = hopefully_kwarg.children[0].value
            else:
                assert isinstance(hopefully_kwarg, Leaf)
                hopefully_kwarg_name = hopefully_kwarg.value
            if hopefully_dstar != _dstar or hopefully_kwarg_name != ast_args.kwarg.arg:
                raise ValueError
        except (IndexError, ValueError):
            raise ValueError(
                f".pyi file expects **{ast_args.kwarg.arg} in source"
            ) from None
        else:
            typedargslist.append(new(_comma))
            typedargslist.append(new(hopefully_dstar))
            typedargslist.append(
                get_annotated_param(
                    hopefully_kwarg, ast_args.kwarg, missing_ok=True, flags=flags
                )
            )

    if params:
        extra_params = minimize_whitespace(
            str(Node(syms.typedargslist, [new(p) for p in params]))
        )
        raise ValueError(f"extra arguments in source: {extra_params}")

    if typedargslist:
        typedargslist = typedargslist[1:]  # drop the initial comma
        if len(typedargslist) == 1:
            # don't pack a single argument to be consistent with how lib2to3
            # parses existing code.
            body = typedargslist[0]
        else:
            body = Node(syms.typedargslist, typedargslist)
        parameters.children = [
            parameters.children[0],  # (
            body,
            parameters.children[-1],  # )
        ]
        for arg in parameters.pre_order():
            # remove now spurious type comments
            arg.prefix = maybe_space_before_comment(
                _type_comment_re.sub(r"\g<nl>", arg.prefix, re.MULTILINE)
            )
    else:
        parameters.children = [
            parameters.children[0],  # (
            parameters.children[-1],  # )
        ]


def annotate_return(function, ast_returns, offset, flags):
    if ast_returns is None:
        if function[offset] == _colon:
            if flags.incremental:
                return

            raise ValueError(
                ".pyi file is missing return value and source doesn't "
                "provide it either"
            )
        elif function[offset] == _rarrow:
            # Source-provided return value, this is fine.
            return

        raise NotImplementedError(f"unexpected return token: {str(function[offset])!r}")

    ret_stmt = convert_annotation(ast_returns)
    ret_stmt.prefix = " "
    if function[offset] == _rarrow:
        function[offset + 1] = maybe_replace_any_if_equal(
            "return value", ret_stmt, function[offset + 1], flags
        )
    elif function[offset] == _colon:
        function.insert(offset, new(_rarrow))
        function.insert(offset + 1, ret_stmt)
    else:
        raise NotImplementedError(f"unexpected return token: {str(function[offset])!r}")


def get_function_signature(fun, *, is_method=False):
    """Returns (args, returns).

    `args` is ast3.arguments, `returns` is the return type AST node. The kicker
    about this function is that it pushes type comments into proper annotation
    fields, standardizing type handling.
    """
    args = fun.args
    returns = fun.returns
    if fun.type_comment:
        try:
            args_tc, returns_tc = parse_signature_type_comment(fun.type_comment)
            if returns and returns_tc:
                raise ValueError(
                    "using both a type annotation and a type comment is not allowed"
                )
            returns = returns_tc
            copy_arguments_to_annotations(args, args_tc, is_method=is_method)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Annotation problem in function {fun.name!r}: "
                + f"{fun.lineno}:{fun.col_offset + 1}: {exc}"
            ) from exc
    copy_type_comments_to_annotations(args)

    return args, returns


def parse_signature_type_comment(type_comment):
    """Parse the fugly signature type comment into AST nodes.

    Caveats: ASTifying **kwargs is impossible with the current grammar so we
    hack it into unary subtraction (to differentiate from Starred in vararg).

    For example from:
    "(str, int, *int, **Any) -> 'SomeReturnType'"

    To:
    ([ast3.Name, ast.Name, ast3.Name, ast.Name], ast3.Str)
    """
    try:
        result = ast3.parse(type_comment, "<func_type>", "func_type")
    except SyntaxError as exc:
        raise ValueError(
            f"invalid function signature type comment: {type_comment!r}"
        ) from exc

    assert isinstance(result, ast3.FunctionType)
    if len(result.argtypes) == 1:
        argtypes = result.argtypes[0]
    else:
        argtypes = result.argtypes
    return argtypes, result.returns


def parse_type_comment(type_comment):
    """Parse a type comment string into AST nodes."""
    try:
        result = ast3.parse(type_comment, "<type_comment>", "eval")
    except SyntaxError:
        raise ValueError(f"invalid type comment: {type_comment!r}") from None

    assert isinstance(result, ast3.Expression)
    return result.body


def parse_arguments(arguments):
    """parse_arguments('(a, b, *, c=False, **d)') -> ast3.arguments

    Parse a string with function arguments into an AST node.
    """
    arguments = f"def f{arguments}: ..."
    try:
        result = ast3.parse(arguments, "<arguments>", "exec")
    except SyntaxError:
        raise ValueError(f"invalid arguments: {arguments!r}") from None

    assert isinstance(result, ast3.Module)
    assert len(result.body) == 1
    assert isinstance(result.body[0], ast3.FunctionDef)
    args = result.body[0].args
    copy_type_comments_to_annotations(args)
    return args


def copy_arguments_to_annotations(args, type_comment, *, is_method=False):
    """Copies AST nodes from `type_comment` into the ast3.arguments in `args`.

    Does validaation of argument count (allowing for untyped self/cls)
    and type (vararg and kwarg).
    """
    if isinstance(type_comment, ast3.Ellipsis):
        return

    expected = len(args.args)
    if args.vararg:
        expected += 1
    expected += len(args.kwonlyargs)
    if args.kwarg:
        expected += 1
    actual = len(type_comment) if isinstance(type_comment, list) else 1
    if expected != actual:
        if is_method and expected - actual == 1:
            pass  # fine, we're just skipping `self`, `cls`, etc.
        else:
            raise ValueError(
                "number of arguments in type comment doesn't match; "
                + f"expected {expected}, found {actual}"
            )

    if isinstance(type_comment, list):
        next_value = type_comment.pop
    else:
        # If there's just one value, only one of the loops and ifs below will
        # be populated. We ensure this with the expected/actual length check
        # above.
        _tc = type_comment

        def next_value(index: int = 0) -> ast3.expr:
            return _tc

    for arg in args.args[expected - actual :]:
        ensure_no_annotation(arg.annotation)
        arg.annotation = next_value(0)

    if args.vararg:
        ensure_no_annotation(args.vararg.annotation)
        args.vararg.annotation = next_value(0)

    for arg in args.kwonlyargs:
        ensure_no_annotation(arg.annotation)
        arg.annotation = next_value(0)

    if args.kwarg:
        ensure_no_annotation(args.kwarg.annotation)
        args.kwarg.annotation = next_value(0)


def copy_type_comments_to_annotations(args):
    """Copies argument type comments from the legacy long form to annotations
    in the entire function signature.
    """
    for arg in args.args:
        copy_type_comment_to_annotation(arg)

    if args.vararg:
        copy_type_comment_to_annotation(args.vararg)

    for arg in args.kwonlyargs:
        copy_type_comment_to_annotation(arg)

    if args.kwarg:
        copy_type_comment_to_annotation(args.kwarg)


def copy_type_comment_to_annotation(arg):
    if not arg.type_comment:
        return

    ann = parse_type_comment(arg.type_comment)
    ensure_no_annotation(arg.annotation)
    arg.annotation = ann


def normalize_strings_to_repr(node):
    """Normalize string leaf nodes to a repr since that is what we generate."""
    if isinstance(node, Leaf) and node.type == token.STRING:
        node.value = repr(ast3.literal_eval(node.value))
    elif isinstance(node, Node):
        node.children = [normalize_strings_to_repr(i) for i in node.children]
    return node


def maybe_replace_any_if_equal(name, expected, actual, flags):
    """Return the type given in `expected`.

    Raise ValueError if `expected` isn't equal to `actual`.  If --replace-any is
    used, the Any type in `actual` is considered equal.

    The implementation is naively checking if the string representation of
    `actual` is one of "Any", "typing.Any", or "t.Any".  This is done for two
    reasons:

    1. I'm lazy.
    2. We want people to be able to explicitly state that they want Any without it
       being replaced.  This way they can use an alias.
    """
    normalized = normalize_strings_to_repr(actual.clone())
    is_equal = expected == normalized
    if not is_equal and flags.replace_any:
        actual_str = minimize_whitespace(str(normalized))
        if actual.type == token.STRING and actual_str:
            actual_str = actual_str[1:-1]
        is_equal = actual_str in {"Any", "typing.Any", "t.Any"}

    if not is_equal:
        expected_annotation = minimize_whitespace(str(expected))
        actual_annotation = minimize_whitespace(str(actual))
        raise ValueError(
            f"incompatible existing {name}. "
            + f"Expected: {expected_annotation!r}, actual: {actual_annotation!r}"
        )

    return expected or actual


def ensure_no_annotation(ann):
    if ann:
        raise ValueError(
            f"using both a type annotation and a type comment is not allowed: {ann}"
        )


def ensure_annotations_equal(name, expected, actual, flags):
    """Raise ValueError if `expected` isn't equal to `actual`.

    If --replace-any is used, the Any type in `actual` is considered equal.
    """
    maybe_replace_any_if_equal(name, expected, actual, flags)


def remove_function_signature_type_comment(body):
    """Removes the legacy signature type comment, leaving other comments if any."""
    for node in body.children:
        if node.type == token.INDENT:
            prefix = node.prefix.lstrip()
            if prefix.startswith("# type: "):
                node.prefix = "\n".join(prefix.split("\n")[1:])
            break


def minimize_whitespace(text):
    return re.sub(r"[\n\t ]+", " ", text, re.MULTILINE).strip()


def maybe_space_before_comment(text):
    if not text:
        return ""

    if text.startswith("#"):
        return "  " + text

    return text


def flatten_some(children):
    """Generates nodes or leaves, unpacking bodies of try:except:finally: statements."""
    for node in children:
        if node.type in (syms.try_stmt, syms.suite):
            yield from flatten_some(node.children)
        else:
            yield node


def pop_param(params):
    """Pops the parameter and the "remainder" (comma, default value).

    Returns a tuple of ('name', default) or (_star, 'name') or (_dstar, 'name').
    """
    default = None

    name = params.pop(0)
    if name in (_star, _dstar):
        default = params.pop(0)
        if default == _comma:
            return name, default

    try:
        remainder = params.pop(0)
        if remainder == _eq:
            default = params.pop(0)
            remainder = params.pop(0)
        if remainder != _comma:
            raise ValueError(f"unexpected token: {remainder}")

    except IndexError:
        pass
    return name, default


def gen_annotated_params(
    args, defaults, params, *, implicit_default=False, is_method=False, flags
):
    missing_ok = is_method or flags.incremental
    for arg, expected_default in zip(args, defaults):
        yield new(_comma)

        try:
            param, actual_default = pop_param(params)
        except IndexError:
            raise ValueError(
                f"missing regular argument {arg.arg!r} in source"
            ) from None

        if param in (_star, _dstar):
            # unexpected *args, keyword-only args, or **kwargs
            raise ValueError(f"missing regular argument {arg.arg!r} in source")

        if expected_default is None and actual_default is not None:
            if not implicit_default or actual_default != _none:
                param_s = minimize_whitespace(str(param))
                raise ValueError(
                    ".pyi file does not specify default value for arg "
                    + f"`{param_s}` but the source does"
                )

        if expected_default is not None and actual_default is None:
            param_s = minimize_whitespace(str(param))
            raise ValueError(
                f"source file does not specify default value for arg `{param_s}` "
                + "but the .pyi file does"
            )

        node = get_annotated_param(param, arg, missing_ok=missing_ok, flags=flags)
        yield node
        if actual_default:
            whitespace = " " if node.type == syms.tname else ""
            yield new(_eq, prefix=whitespace)
            yield new(actual_default, prefix=whitespace)

        missing_ok = flags.incremental


def get_annotated_param(node, arg, *, missing_ok=False, flags):
    if node.type not in (token.NAME, syms.tname):
        raise NotImplementedError(f"unexpected node token: `{node}`")

    actual_ann = None
    if node.type == syms.tname:
        actual_ann = node.children[2]
        node = node.children[0]
    if not isinstance(node, Leaf) or arg.arg != node.value:
        raise ValueError(
            f".pyi file expects argument {arg.arg!r} next but argument "
            + f"{minimize_whitespace(str(node))!r} found in source"
        )

    if arg.annotation is None:
        if actual_ann is None:
            if missing_ok:
                return new(node)

            raise ValueError(
                f".pyi file is missing annotation for {arg.arg!r} and source "
                + "doesn't provide it either"
            )

        ann = new(actual_ann)
    else:
        ann = convert_annotation(arg.annotation)
        ann.prefix = " "

    if actual_ann is not None:
        ensure_annotations_equal(
            f"annotation for {arg.arg!r}", ann, actual_ann, flags=flags
        )

    return Node(syms.tname, [new(node), new(_colon), ann])


def get_offset_and_prefix(body, skip_assignments=False):
    """Returns the offset after which a statement can be inserted to the `body`.

    This offset is calculated to come after all imports, and maybe existing
    (possibly annotated) assignments if `skip_assignments` is True.

    Also returns the indentation prefix that should be applied to the inserted
    node.
    """
    assert body.type in (syms.file_input, syms.suite)

    _offset = 0
    prefix = ""
    for _offset, child in enumerate(body.children):
        if child.type == syms.simple_stmt:
            stmt = child.children[0]
            if stmt.type == syms.expr_stmt:
                expr = stmt.children
                if not skip_assignments:
                    break

                if (
                    len(expr) != 2
                    or expr[0].type != token.NAME
                    or expr[1].type != syms.annassign
                    or _eq in expr[1].children
                ):
                    break

            elif stmt.type not in (syms.import_name, syms.import_from, token.STRING):
                break

        elif child.type == token.INDENT:
            assert isinstance(child, Leaf)
            prefix = child.value
        elif child.type != token.NEWLINE:
            break

    prefix, child.prefix = child.prefix, prefix
    return _offset, prefix


@singledispatch
def name_used_in_node(node, name):
    """Returns True if `name` appears in `node`. False otherwise."""


@name_used_in_node.register(Node)
def _nuin_node(node, name):
    for n in node.pre_order():
        if n == name:
            return True

    return False


@name_used_in_node.register(Leaf)
def _nuin_leaf(leaf, name):
    return leaf == name


def fix_line_numbers(body):
    r"""Recomputes all line numbers based on the number of \n characters."""
    maxline = 0
    for node in body.pre_order():
        maxline += node.prefix.count("\n")
        if isinstance(node, Leaf):
            node.lineno = maxline
            maxline += str(node.value).count("\n")


def new(n, prefix=None):
    """lib2to3's AST requires unique objects as children."""

    if isinstance(n, Leaf):
        return Leaf(n.type, n.value, prefix=n.prefix if prefix is None else prefix)

    # this is hacky, we assume complex nodes are just being reused once from the
    # original AST.
    n.parent = None
    if prefix is not None:
        n.prefix = prefix
    return n


def _load_ignore(at_path, parent_spec, ignores):
    ignore_file = at_path / ".gitignore"
    if not ignore_file.exists():
        return parent_spec
    lines = ignore_file.read_text().split(os.linesep)
    spec = PathSpec.from_lines("gitwildmatch", lines)
    spec = PathSpec(parent_spec.patterns + spec.patterns)
    ignores[at_path] = spec
    return spec


def walk_not_git_ignored(path, keep, extra_ignore):
    spec = PathSpec.from_lines("gitwildmatch", [".git"] + extra_ignore)
    ignores = {}
    # detect git folder, collect ignores up to root
    at = path
    while True:
        git_exist = (at / ".git").exists()
        if git_exist:
            ignores[at.parent] = spec
            # go down back and load all ignores
            for part in (".",) + path.relative_to(at).parts:
                at = at / part
                spec = _load_ignore(at, spec, ignores)
            break
        if at == at.parent:
            ignores[path] = spec
            break
        at = at.parent

    # now walk from root, collect new ignores and evaluate
    for root, dirs, files in os.walk(str(path)):
        root_path = Path(root).relative_to(path)  # current path
        current_path = path / root
        parent_spec = ignores.get(current_path) or next(
            ignores[p] for p in current_path.parents if p in ignores
        )
        spec = _load_ignore(Path(root), parent_spec, ignores)
        for file_name in files:
            result = root_path / file_name
            if (
                file_name != ".gitignore"
                and keep(result)
                and not spec.match_file(str(result))
            ):
                yield path / result
        for cur_dir in list(dirs):
            if spec.match_file(str(root_path / cur_dir)):
                dirs.remove(cur_dir)


_as = Leaf(token.NAME, "as", prefix=" ")
_colon = Leaf(token.COLON, ":")
_comma = Leaf(token.COMMA, ",")
_dot = Leaf(token.DOT, ".")
_dstar = Leaf(token.DOUBLESTAR, "**")
_eq = Leaf(token.EQUAL, "=", prefix=" ")
_lpar = Leaf(token.LPAR, "(")
_lsqb = Leaf(token.LSQB, "[")
_newline = Leaf(token.NEWLINE, "\n")
_none = Leaf(token.NAME, "None")
_rarrow = Leaf(token.RARROW, "->", prefix=" ")
_rpar = Leaf(token.RPAR, ")")
_rsqb = Leaf(token.RSQB, "]")
_star = Leaf(token.STAR, "*")
_ellipsis = Node(syms.atom, children=[new(_dot), new(_dot), new(_dot)])

_type_comment_re = re.compile(
    r"""
    ^
    [\t ]*
    \#[ ]type:[ ]*
    (?P<type>
        [^#\t\n]+?
    )
    (?<!ignore)     # note: this will force the non-greedy + in <type> to match
                    # a trailing space which is why we need the silliness below
    (?<!ignore[ ]{1})(?<!ignore[ ]{2})(?<!ignore[ ]{3})(?<!ignore[ ]{4})
    (?<!ignore[ ]{5})(?<!ignore[ ]{6})(?<!ignore[ ]{7})(?<!ignore[ ]{8})
    (?<!ignore[ ]{9})(?<!ignore[ ]{10})
    [\t ]*
    (?P<nl>
        (?:\#[^\n]*)?
        \n?
    )
    $
    """,
    re.MULTILINE | re.VERBOSE,
)


__all__ = [
    "__version__",
    "retype_path",
    "retype_file",
    "ReApplyFlags",
]
