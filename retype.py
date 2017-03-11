#!/usr/bin/env python3
"""Re-apply type annotations from .pyi stubs to your codebase."""

from __future__ import print_function

from functools import partial, singledispatch
from lib2to3 import pygram, pytree
from lib2to3.pgen2 import driver
from lib2to3.pgen2 import token
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import _detect_future_features
from lib2to3.pygram import python_symbols as syms
from lib2to3.pytree import Node, Leaf
from pathlib import Path
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
                yield (child, e)
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
    for node in pyi_ast.body:
        reapply(node, src_node)
    targets.mkdir(parents=True, exist_ok=True)
    with open(targets / py, 'w') as target_file:
        target_file.write(lib2to3_unparse(src_node))
    return targets / py


def lib2to3_parse(src_txt):
    """Given a string with source, return the lib2to3 Node."""
    features = _detect_future_features(src_txt)
    grammar = pygram.python_grammar
    if 'print_function' in features:
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


@singledispatch
def reapply(ast_node, lib2to3_node):
    """Reapplies the typed_ast node into the lib2to3 tree.

    By default does nothing.
    """


@reapply.register(ast3.ImportFrom)
def _r_importfrom(import_from, node):
    assert node.type == syms.file_input
    level = import_from.level
    module = '.' * level + import_from.module
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


@reapply.register(ast3.Import)
def _r_import(import_, node):
    assert node.type == syms.file_input
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


@singledispatch
def names_already_imported(names, node):
    """Returns True if `node` represents `names`."""
    return False


@names_already_imported.register(list)
def _nai_list(names, node):
    return all(names_already_imported(name, node) for name in names)


_as = Leaf(token.NAME, 'as', prefix=" ")


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


def issublist(sublist, superlist):
    n = len(sublist)
    return any((sublist == superlist[i:i + n]) for i in range(len(superlist) - n + 1))


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
    assert node.type == syms.file_input
    insert_after = -1
    for index, child in enumerate(node.children):
        if child.type != syms.simple_stmt:
            continue

        stmt = child.children[0]
        if stmt.type in (syms.import_name, syms.import_from, token.STRING):
            insert_after = index

    node.children.insert(insert_after + 1, stmt_to_insert)


if __name__ == '__main__':
    main()
