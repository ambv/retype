#!/usr/bin/env python3
"""Re-apply type annotations from .pyi stubs to your codebase."""

from functools import partial, singledispatch
from lib2to3 import pygram, pytree
from lib2to3.pgen2 import driver
from lib2to3.pgen2 import token
from lib2to3.pgen2.parse import ParseError
from lib2to3.pygram import python_symbols as syms
from lib2to3.pytree import Node, Leaf, type_repr
from pathlib import Path
import re
import sys

import click
from typed_ast import ast3

__version__ = "17.3.0"

Directory = partial(
    click.Path,
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    writable=False,
)


@click.command()
@click.option(
    '--src-dir',
    type=Directory(),
    default='.',
    help='Where to find sources.',
    show_default=True
)
@click.option(
    '--pyi-dir',
    type=Directory(),
    default='types',
    help='Where to find .pyi stubs.',
    show_default=True,
)
@click.option(
    '--target-dir',
    type=Directory(exists=False, writable=True),
    default='typed-src',
    help='Where to write annotated sources.',
    show_default=True,
)
@click.version_option(version=__version__)
def main(src_dir, pyi_dir, target_dir):
    """Re-apply type annotations from .pyi stubs to your codebase."""
    returncode = 0
    for file, error in retype_path(
        Path(pyi_dir),
        srcs=Path(src_dir),
        targets=Path(target_dir),
    ):
        print(f'error: {file}: {error}', file=sys.stderr)
        returncode += 1

    # According to http://tldp.org/LDP/abs/html/index.html starting with 126
    # we have special returncodes.
    sys.exit(min(returncode, 125))


def retype_path(path, srcs, targets):
    """Recursively retype files in the given directories. Generate errors."""
    for child in path.iterdir():
        if child.is_dir():
            yield from retype_path(child, srcs / child.name, targets / child.name)
        elif child.suffix == '.pyi':
            try:
                retype_file(child, srcs, targets)
            except Exception as e:
                yield (child, str(e))
        elif child.name.startswith('.'):
            continue  # silently ignore dot files.
        else:
            yield (child, f'Unexpected file in {path}')


def retype_file(pyi, srcs, targets):
    """Based on `pyi`, find a file stored in `srcs`, retype, save in `targets`.

    The file should remain formatted exactly as it was before, save for:
    - annotations
    - additional imports needed to satisfy annotations
    - additional module-level names needed to satisfy annotations
    """
    with open(pyi) as pyi_file:
        pyi_txt = pyi_file.read()
    py = pyi.name[:-1]
    with open(srcs / py) as src_file:
        src_txt = src_file.read()
    pyi_ast = ast3.parse(pyi_txt)
    src_node = lib2to3_parse(src_txt)
    reapply_all(pyi_ast.body, src_node)
    targets.mkdir(parents=True, exist_ok=True)
    with open(targets / py, 'w') as target_file:
        target_file.write(lib2to3_unparse(src_node))
    return targets / py


def lib2to3_parse(src_txt):
    """Given a string with source, return the lib2to3 Node."""
    grammar = pygram.python_grammar_no_print_statement
    drv = driver.Driver(grammar, pytree.convert)
    try:
        result = drv.parse_string(src_txt, True)
    except ParseError as pe:
        lineno, column = pe.context[1]
        faulty_line = src_txt.splitlines()[lineno - 1]
        raise ValueError(f"Cannot parse: {lineno}:{column}: {faulty_line}") from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])

    return result


def lib2to3_unparse(node):
    """Given a lib2to3 node, return its string representation."""
    return str(node)


def reapply_all(ast_node, lib2to3_node):
    """Reapplies the typed_ast node into the lib2to3 tree.

    Also does post-processing. This is done in reverse order to enable placing
    TypeVars and aliases that depend on one another.
    """
    late_processing = reapply(ast_node, lib2to3_node)
    for lazy_func in reversed(late_processing):
        lazy_func()


@singledispatch
def reapply(ast_node, lib2to3_node):
    """Reapplies the typed_ast node into the lib2to3 tree.

    By default does nothing.
    """
    return []


@reapply.register(list)
def _r_list(l, lib2to3_node):
    if lib2to3_node.type not in (syms.file_input, syms.suite):
        return []

    result = []
    for pyi_node in l:
        result.extend(reapply(pyi_node, lib2to3_node))
    return result


@reapply.register(ast3.ImportFrom)
def _r_importfrom(import_from, node):
    assert node.type in (syms.file_input, syms.suite)
    level = import_from.level or 0
    module = '.' * level + (import_from.module or '')
    names = import_from.names
    for child in node.children:
        if child.type != syms.simple_stmt:
            continue

        stmt = child.children[0]
        if stmt.type == syms.import_from:
            imp = stmt.children
            # if the module we're looking for is already imported, skip it.
            if str(imp[1]).strip() == module and names_already_imported(names, imp[3]):
                break
    else:
        imp = make_import(*names, from_module=module)
        append_after_imports(imp, node)
    return []


@reapply.register(ast3.Import)
def _r_import(import_, node):
    assert node.type in (syms.file_input, syms.suite)
    names = import_.names
    for child in node.children:
        if child.type != syms.simple_stmt:
            continue

        stmt = child.children[0]
        if stmt.type == syms.import_name:
            imp = stmt.children
            # if the module we're looking for is already imported, skip it.
            if names_already_imported(names, imp[1]):
                break
    else:
        imp = make_import(*names)
        append_after_imports(imp, node)
    return []


@reapply.register(ast3.ClassDef)
def _r_classdef(cls, node):
    assert node.type in (syms.file_input, syms.suite)
    name = Leaf(token.NAME, cls.name)
    for child in node.children:
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
        result.extend(reapply(ast_elem, cls_node))
    return result


@reapply.register(ast3.FunctionDef)
def _r_functiondef(fun, node):
    assert node.type in (syms.file_input, syms.suite)
    name = Leaf(token.NAME, fun.name)
    args = fun.args
    returns = fun.returns
    is_method = node.parent is not None and node.parent.type == syms.classdef
    for child in node.children:
        decorators = None
        if child.type == syms.decorated:
            # skip decorators
            decorators = child.children[0]
            child = child.children[1]

        if child.type == syms.funcdef:
            offset = 1
        elif child.type == syms.async_funcdef:
            offset = 2
        else:
            continue

        if child.children[offset] == name:
            lineno = child.get_lineno()
            column = 1

            if is_method and decorators:
                pyi_decorators = decorator_names(fun.decorator_list)
                src_decorators = decorator_names(decorators)
                pyi_builtin_decorators = list(
                    filter(is_builtin_method_decorator, pyi_decorators)
                ) or ['instancemethod']
                src_builtin_decorators = list(
                    filter(is_builtin_method_decorator, src_decorators)
                ) or ['instancemethod']
                if pyi_builtin_decorators != src_builtin_decorators:
                    raise ValueError(
                        f"Incompatible method kind for {fun.name!r}: " +
                        f"{lineno}:{column}: Expected: " +
                        f"{pyi_builtin_decorators[0]}, actual: " +
                        f"{src_builtin_decorators[0]}"
                    )

                is_method = "staticmethod" not in pyi_decorators

            try:
                annotate_parameters(
                    child.children[offset + 1], args, is_method=is_method
                )
                annotate_return(child.children, returns, offset + 2)
                reapply(fun.body, child.children[-1])
            except ValueError as ve:
                raise ValueError(
                    f"Annotation problem in function {name.value!r}: " +
                    f"{lineno}:{column}: {ve}"
                )
            break
    else:
        raise ValueError(f"Function {name.value!r} not found in source.")

    return []


@reapply.register(ast3.AnnAssign)
def _r_annassign(annassign, body):
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
    annassign_node = Node(
        syms.annassign,
        [
            new(_colon),
            annotation,
        ],
    )
    for child in body.children:
        if child.type != syms.simple_stmt:
            continue

        maybe_expr = child.children[0]
        if maybe_expr.type != syms.expr_stmt:
            continue

        expr = maybe_expr.children
        maybe_annotation = None

        if (
            expr[0].type in (token.NAME, syms.power) and
            minimize_whitespace(str(expr[0])) == name
        ):
            if expr[1].type == syms.annassign:
                # variable already typed
                maybe_annotation = expr[1].children[1]
                if len(expr[1].children) > 2 and expr[1].children[2] != _eq:
                    raise NotImplementedError(
                        f"unexpected element after annotation: {str(expr[3])}"
                    )
            elif expr[1] != _eq:
                # If it's not an assignment, we're ignoring it. It could be:
                # - indexing
                # - tuple unpacking
                # - calls
                # - etc. etc.
                continue

            if maybe_annotation is not None:
                if annotation != maybe_annotation:
                    expected_annotation = minimize_whitespace(str(annotation))
                    actual_annotation = minimize_whitespace(str(maybe_annotation))
                    raise ValueError(
                        f"incompatible existing variable annotation for " +
                        f"{name!r}. Expected: " +
                        f"{expected_annotation!r}, actual: {actual_annotation!r}"
                    )
            else:
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
                    Node(
                        syms.expr_stmt,
                        [
                            Leaf(token.NAME, name),
                            annassign_node,
                        ],
                    ),
                    new(_newline),
                ],
                prefix=prefix.lstrip('\n'),
            ),
        )

    return []


@reapply.register(ast3.Assign)
def _r_assign(assign, body):
    assert body.type in (syms.file_input, syms.suite)

    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast3.Name):
        # Type aliases cannot have multiple targets, and cannot be attributes.
        # We're not interested in other assignments.
        return []

    name = assign.targets[0].id
    value = convert_annotation(assign.value)
    value.prefix = " "

    for child in body.children:
        if child.type != syms.simple_stmt:
            continue

        maybe_expr = child.children[0]
        if maybe_expr.type != syms.expr_stmt:
            continue

        expr = maybe_expr.children

        if (expr[0].type == token.NAME and expr[0].value == name and expr[1] == _eq):
            actual_value = expr[2]
            if value != actual_value:
                value_str = minimize_whitespace(str(value))
                actual_value_str = minimize_whitespace(str(actual_value))
                raise ValueError(
                    f"incompatible existing alias {name!r}. Expected: " +
                    f"{value_str!r}, actual: {actual_value_str!r}"
                )

            break
    else:
        # We need to defer placing aliases because we need to place them
        # relative to their usage, and the type annotations likely come after
        # in the .pyi file.

        def lazy_aliasing():
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
                        Node(
                            syms.expr_stmt,
                            [
                                Leaf(token.NAME, name),
                                new(_eq),
                                value,
                            ],
                        ),
                        new(_newline),
                    ],
                    prefix=prefix.lstrip('\n'),
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


@singledispatch
def convert_annotation(ann):
    """Converts an AST object into its lib2to3 equivalent."""
    raise NotImplementedError(f"unknown AST node type: {ann!r}")


@convert_annotation.register(ast3.Subscript)
def _c_subscript(sub):
    return Node(
        syms.power,
        [
            convert_annotation(sub.value), Node(
                syms.trailer,
                [
                    new(_lsqb),
                    convert_annotation(sub.slice),
                    new(_rsqb),
                ],
            )
        ],
    )


@convert_annotation.register(ast3.Name)
def _c_name(name):
    return Leaf(token.NAME, name.id)


@convert_annotation.register(ast3.NameConstant)
def _c_nameconstant(const):
    return Leaf(token.NAME, repr(const.value))


@convert_annotation.register(ast3.Ellipsis)
def _c_ellipsis(ell):
    return Node(syms.atom, [new(_dot), new(_dot), new(_dot)])


@convert_annotation.register(ast3.Str)
def _c_str(s):
    return Leaf(token.STRING, repr(s.s))


@convert_annotation.register(ast3.Index)
def _c_index(index):
    return convert_annotation(index.value)


@convert_annotation.register(ast3.Tuple)
def _c_tuple(tup):
    contents = [convert_annotation(elt) for elt in tup.elts]
    for index in range(len(contents) - 1, 0, -1):
        contents[index].prefix = " "
        contents.insert(index, new(_comma))

    return Node(
        syms.subscriptlist,
        contents,
    )


@convert_annotation.register(ast3.Attribute)
def _c_attribute(attr):
    # This is hacky. ¯\_(ツ)_/¯
    return Leaf(token.NAME, f"{convert_annotation(attr.value)}.{attr.attr}")


@convert_annotation.register(ast3.Call)
def _c_call(call):
    contents = [convert_annotation(arg) for arg in call.args]
    contents.extend(convert_annotation(kwarg) for kwarg in call.keywords)
    for index in range(len(contents) - 1, 0, -1):
        contents[index].prefix = " "
        contents.insert(index, new(_comma))

    call_args = [
        new(_lpar),
        new(_rpar),
    ]
    if contents:
        call_args.insert(1, Node(syms.arglist, contents))
    return Node(
        syms.power,
        [convert_annotation(call.func), Node(
            syms.trailer,
            call_args,
        )],
    )


@convert_annotation.register(ast3.keyword)
def _c_keyword(kwarg):
    return Node(
        syms.argument,
        [
            Leaf(token.NAME, kwarg.arg),
            new(_eq, prefix=''),
            convert_annotation(kwarg.value),
        ],
    )


@convert_annotation.register(ast3.List)
def _c_list(l):
    contents = [convert_annotation(elt) for elt in l.elts]
    for index in range(len(contents) - 1, 0, -1):
        contents[index].prefix = " "
        contents.insert(index, new(_comma))

    list_literal = [
        new(_lsqb),
        new(_rsqb),
    ]
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


def is_builtin_method_decorator(name):
    return name in {'classmethod', 'staticmethod'}


def make_import(*names, from_module=None):
    assert names
    imports = []

    if from_module:
        statement = syms.import_from
        container = syms.import_as_names
        single = syms.import_as_name
        result = [
            Leaf(token.NAME, 'from'),
            Leaf(token.NAME, from_module, prefix=' '),
            Leaf(token.NAME, 'import', prefix=' '),
        ]
    else:
        statement = syms.import_name
        container = syms.dotted_as_names
        single = syms.dotted_as_name
        result = [Leaf(token.NAME, 'import')]

    for alias in names:
        name = Leaf(token.NAME, alias.name, prefix=' ')
        if alias.asname:
            _as = Leaf(token.NAME, 'as', prefix=' ')
            asname = Leaf(token.NAME, alias.asname, prefix=' ')
            imports.append(Node(single, [name, _as, asname]))
        else:
            imports.append(name)
    if len(imports) == 1:
        result.append(imports[0])
    else:
        imports_and_commas = []
        for imp in imports[:-1]:
            imports_and_commas.append(imp)
            imports_and_commas.append(Leaf(token.COMMA, ','))
        imports_and_commas.append(imports[-1])
        result.append(Node(container, imports_and_commas))
    return Node(
        syms.simple_stmt,
        [
            Node(statement, result),
            Leaf(token.NEWLINE, '\n'),  # FIXME: \r\n?
        ],
    )


def append_after_imports(stmt_to_insert, node):
    offset, stmt_to_insert.prefix = get_offset_and_prefix(node)
    node.children.insert(offset, stmt_to_insert)


def annotate_parameters(parameters, ast_args, *, is_method=False):
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
    defaults = [None] * num_args_no_defaults + ast_args.defaults
    typedargslist.extend(
        gen_annotated_params(ast_args.args, defaults, params, is_method=is_method)
    )
    if ast_args.vararg or ast_args.kwonlyargs:
        try:
            hopefully_star, hopefully_vararg = pop_param(params)
            if hopefully_star != _star:
                raise ValueError
        except (IndexError, ValueError):
            raise ValueError(
                f".pyi file expects *args or keyword-only arguments in source"
            ) from None
        else:
            typedargslist.append(new(_comma))
            typedargslist.append(new(hopefully_star))

    if ast_args.vararg:
        if hopefully_vararg.type == syms.tname:
            hopefully_vararg_name = hopefully_vararg.children[0].value
        else:
            hopefully_vararg_name = hopefully_vararg.value
        if hopefully_vararg_name != ast_args.vararg.arg:
            raise ValueError(f".pyi file expects *{ast_args.vararg.arg} in source")

        typedargslist.append(
            get_annotated_param(hopefully_vararg, ast_args.vararg, missing_ok=True)
        )

    if ast_args.kwonlyargs:
        if not ast_args.vararg:
            if hopefully_vararg != _comma:
                raise ValueError(
                    f".pyi file expects keyword-only arguments but " +
                    f"*{str(hopefully_vararg).strip()} found in source"
                )

        typedargslist.extend(
            gen_annotated_params(
                ast_args.kwonlyargs,
                ast_args.kw_defaults,
                params,
                implicit_default=True,
            )
        )

    if ast_args.kwarg:
        try:
            hopefully_dstar, hopefully_kwarg = pop_param(params)
            if hopefully_kwarg.type == syms.tname:
                hopefully_kwarg_name = hopefully_kwarg.children[0].value
            else:
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
                get_annotated_param(hopefully_kwarg, ast_args.kwarg, missing_ok=True)
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
    else:
        parameters.children = [
            parameters.children[0],  # (
            parameters.children[-1],  # )
        ]


def annotate_return(function, ast_returns, offset):
    if ast_returns is None:
        if function[offset] == _colon:
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
        existing_return = function[offset + 1]
        if existing_return != ret_stmt:
            ret_stmt_str = minimize_whitespace(str(ret_stmt))
            existing_return_str = minimize_whitespace(str(existing_return))
            raise ValueError(
                f"incompatible existing return value. Expected: " +
                f"{ret_stmt_str!r}, actual: {existing_return_str!r}"
            )
    elif function[offset] == _colon:
        function.insert(offset, new(_rarrow))
        function.insert(offset + 1, ret_stmt)
    else:
        raise NotImplementedError(f"unexpected return token: {str(function[offset])!r}")


def minimize_whitespace(text):
    return re.sub(r'[\n\t ]+', ' ', text, re.MULTILINE).strip()


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
    args, defaults, params, *, implicit_default=False, is_method=False
):
    missing_ok = is_method
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
                raise ValueError(
                    f".pyi file does not specify default value for arg " +
                    f"`{param.value}` but the source does"
                )

        if expected_default is not None and actual_default is None:
            raise ValueError(
                f"source file does not specify default value for arg `{param.value}` " +
                f"but the .pyi file does"
            )

        yield get_annotated_param(param, arg, missing_ok=missing_ok)
        if actual_default:
            yield new(_eq)
            yield new(actual_default, prefix=' ')

        missing_ok = False


def get_annotated_param(node, arg, *, missing_ok=False):
    if node.type not in (token.NAME, syms.tname):
        raise NotImplementedError(f"unexpected node token: `{node}`")

    actual_ann = None
    if node.type == syms.tname:
        actual_ann = node.children[2]
        node = node.children[0]
    if arg.arg != node.value:
        raise ValueError(
            f".pyi file expects argument {arg.arg!r} next but argument " +
            f"{node.value!r} found in source"
        )

    if arg.annotation is None and actual_ann is None:
        if missing_ok:
            return new(node)

        raise ValueError(
            f".pyi file is missing annotation for {arg.arg!r} and source " +
            f"doesn't provide it either"
        )

    if arg.annotation is None:
        ann = new(actual_ann)
    else:
        ann = convert_annotation(arg.annotation)
        ann.prefix = ' '

    if actual_ann is None:
        actual_ann = ann

    if ann != actual_ann:
        ann_str = minimize_whitespace(str(ann))
        actual_ann_str = minimize_whitespace(str(actual_ann))
        raise ValueError(
            f"incompatible annotation for {arg.arg!r}. Expected: " +
            f"{ann_str!r}, actual: {actual_ann_str!r}"
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
    prefix = ''
    for _offset, child in enumerate(body.children):
        if child.type == syms.simple_stmt:
            stmt = child.children[0]
            if stmt.type == syms.expr_stmt:
                expr = stmt.children
                if not skip_assignments:
                    break

                if (
                    len(expr) != 2 or
                    expr[0].type != token.NAME or
                    expr[1].type != syms.annassign or
                    _eq in expr[1].children
                ):
                    break

            elif stmt.type not in (syms.import_name, syms.import_from, token.STRING):
                break

        elif child.type == token.INDENT:
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
        maxline += node.prefix.count('\n')
        if isinstance(node, Leaf):
            node.lineno = maxline
            maxline += str(node.value).count('\n')


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


_as = Leaf(token.NAME, 'as', prefix=' ')
_colon = Leaf(token.COLON, ':')
_comma = Leaf(token.COMMA, ',')
_dot = Leaf(token.DOT, '.')
_dstar = Leaf(token.DOUBLESTAR, '**')
_eq = Leaf(token.EQUAL, '=', prefix=' ')
_lpar = Leaf(token.LPAR, '(')
_lsqb = Leaf(token.LSQB, '[')
_newline = Leaf(token.NEWLINE, '\n')
_none = Leaf(token.NAME, 'None')
_rarrow = Leaf(token.RARROW, '->', prefix=' ')
_rpar = Leaf(token.RPAR, ')')
_rsqb = Leaf(token.RSQB, ']')
_star = Leaf(token.STAR, '*')

if __name__ == '__main__':
    main()
