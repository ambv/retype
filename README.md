# retype

[![Latest version on
PyPi](https://badge.fury.io/py/retype.svg)](https://badge.fury.io/py/retype)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/retype.svg)](https://pypi.org/project/retype/)
![check](https://github.com/ambv/retype/workflows/check/badge.svg)
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/retype/month)](https://pepy.tech/project/retype/month)

Re-apply type annotations from .pyi stubs to your codebase.

## Usage

```
Usage: retype [OPTIONS] [SRC]...

  Re-apply type annotations from .pyi stubs to your codebase.

Options:
  -p, --pyi-dir DIRECTORY     Where to find .pyi stubs.  [default: types]
  -t, --target-dir DIRECTORY  Where to write annotated sources.  [default:
                              typed-src]
  -i, --incremental           Allow for missing type annotations in both stubs
                              and the source.
  -q, --quiet                 Don't emit warnings, just errors.
  -a, --replace-any           Allow replacing Any annotations.
  --hg                        Post-process files to preserve implicit byte
                              literals.
  --traceback                 Show a Python traceback on error.
  --version                   Show the version and exit.
  --help                      Show this message and exit.
```

When you run `retype`, it goes through all files you passed as SRC,
finds the corresponding .pyi files in the `types/` directory, and
re-applies typing annotations from .pyi to the sources, using the
Python 3 function and variable annotation syntax.  The resulting
combined sources are saved in `typed-src/`.

You can also pass directories as sources, in which case `retype` will
look for .py files in them recursively.

It's smart enough to do the following:

* reapply typing imports
* reapply function argument annotations
* reapply function return value annotations
* reapply method argument and return value annotations
* reapply function-level variable annotations
* reapply module-level name annotations
* reapply module-level type aliases
* reapply class-level field annotations
* reapply instance-level field annotations
* validate existing source annotations against the .pyi file
* validate source function signatures against the .pyi file
* read function signature type comments in .pyi files
* read variable type comments in .pyi files
* consider existing source type comments as annotations
* remove duplicate type comments from source when annotations are applied
* normalize remaining type comments in the source to annotations; this
  is done even if the corresponding .pyi file is missing


## List of things to be done

* [ ] add a --backward option to output type comments instead of annotations
* [ ] handle if sys.version_info and sys.platform checks in stubs


## Design principles

* it's okay for a given .pyi file to be incomplete (gradual typing,
  baby!)
* it's okay for functions and classes to be out of order in .pyi files
  and the source
* it's an **error** for a function or class to be missing in the source
* it's an **error** for a function's signature to be incompatible
  between the .pyi file and the source
* it's an **error** for an annotation in the source to be incompatible
  with the .pyi file


## Known limitations

* Line numbers in the annotated source will no longer match original
  source code; this is because re-application of types requires copying
  typing imports and alias definitions from the .pyi file.
* While formatting of the original source will be preserved, formatting
  of the applied annotations might differ from the formatting in .pyi
  files.
* The source where type annotations get re-applied cannot use the
  legacy `print` statement; that wouldn't work at runtime.
* Class attribute annotations in `__init__()` methods are moved verbatim
  to the respective `__init__()` method in the implementation.  They are
  never translated into class-level attribute annotations, so if that
  method is missing, the translation will fail.  Similarly, class-level
  attribute annotations are never applied to `__init__()` methods.
* Forward references in .pyi files will only be properly resolved for
  type aliases and type vars (by inserting them right before they're
  used in the source).  Other forms of forward references will not work
  in the source code due to out-of-order class and function definitions.
  Modify your .pyi files to use strings.  `retype` will not
  automatically discover failing forward references and stringify them.
* Local variable annotations present in the .pyi file are transferred to
  the body level of the given function in the source.  In other words,
  if the source defines a variable within a loop or a conditional
  statement branch, `retype` will create an value-less variable
  annotation at the beginning of the function.  Use a broad type and
  constrain types in relevant code paths using `assert isinstance()`
  checks.
* Because of the above, existing source variable annotations and type
  comments buried in conditionals and loops will not be deduplicated
  (and `mypy` will complain that a name was already defined).
* An async function in the stub will match a regular function of the
  same name in the same scope and vice versa.  This is to enable
  annotating async functions spelled with `@asyncio.coroutine`.


## Tests

Just run:

```
tox
```

## OMG, this is Python 3 only!

Relax, you can run *retype* **as a tool** perfectly fine under Python
3.6+ even if you want to analyze Python 2 code.  This way you'll be able
to parse all of the new syntax supported on Python 3 but also
*effectively all* the Python 2 syntax at the same time.

By making the code exclusively Python 3.6+, I'm able to focus on the
quality of the checks and re-use all the nice features of the new
releases (check out [pathlib](docs.python.org/3/library/pathlib.html)
or f-strings) instead of wasting cycles on Unicode compatibility, etc.

Note: to retype modules using f-strings you need to run on Python 3.6.2+
due to [bpo-23894](http://bugs.python.org/issue23894).

## License

MIT


## Change Log

### 19.9.0

* add a module entry-point, now you can call it via ``python -m retype``
* automatically all files excluded by ``.gitignore`` on merge of folders
* support for ``ast3.num``
* fix a bug that meant the merge was not recursive in paths
* use `setup.cfg` based packaging configuration
* add PEP-517/8 declaration via `pyproject.toml`
* include license in both wheel and sdist
* this projects code base is now formatted with *black*, import ordered via
  *isort*, and uses Azure Pipelines instead of Travis (also testing on Windows
  and macOs)

### 17.12.0

* support --replace-any to allow replacing pre-existing `Any` annotations
  without raising errors

* bugfix: don't re-apply `# type: ignore` as an annotation if followed
  by another comment.  Original patch by Shannon Zhu.

### 17.6.3

* bugfix: don't try to re-apply `# type: ignore` as a function annotation

* bugfix: support arbitrary source file encodings, patch by Michael Overmeyer.

* bugfix: support missing newlines at the end of the file, patch by Michael
  Overmeyer.

* bugfix: in --incremental, format default values according to PEP 8
  (no spaces around the = sign if the type is missing)

### 17.6.2

* bugfix: --incremental didn't work with multiple arguments before

### 17.6.1

* support --incremental stub application (i.e. allow for both stubs and the
  source to be missing annotations for some arguments and/or return value)

### 17.6.0

* support async functions

* support --traceback for getting more information about internal errors

### 17.4.0

* first published version

* date-versioned


## Authors

Glued together by [Łukasz Langa](mailto:lukasz@langa.pl).  Multiple
improvements by [Michael Overmeyer](mailto:m.overmeyer@yahoo.ca) and
[Bernat Gabor](mailto:gaborjbernat@gmail.com).
