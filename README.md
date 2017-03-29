# retype

[![Build Status](https://travis-ci.org/ambv/retype.svg?branch=master)](https://travis-ci.org/ambv/retype)

Re-apply type annotations from .pyi stubs to your codebase.

## Usage

When you run `retype`, it will look for .pyi files in the `types/`
directory, and for each such file, it will re-apply typing annotations
to respective source files found in the current working directory.
The resulting combined sources are put in `typed-src/`.

All those directories are customizable, see `--help`.


## This is a work in progress.

Things to be done:

* [x] reapply typing imports
* [x] reapply function argument annotations
* [x] reapply function return value annotations
* [x] reapply method argument and return value annotations
* [x] reapply function-level variable annotations
* [x] reapply module-level field annotations
* [x] reapply module-level type aliases
* [x] reapply class-level field annotations
* [x] reapply instance-level field annotations
* [ ] add the --keep-byte-literals option for Mercurial
* [ ] support type comments in .pyi files
* [ ] support type ignore comments in .pyi files
* [ ] add a --backward option to output type comments instead of annotations
* [ ] handle if sys.version_info and sys.platform checks in stubs
* [ ] warn about functions and classes with missing annotations


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
  type aliases (by inserting them right before they're used in the
  source).  Other forms of forward references will not work in the
  source code due to out-of-order class and function definitions.
  Modify your .pyi files to use strings.  `retype` will not automatically
  discover failing forward references and stringify them.


## Tests

Just run:

```
python setup.py test
```

## OMG, this is Python 3 only!

Relax, you can run *retype* **as a tool** perfectly fine under Python
3.6+ even if you want to analyze Python 2 code.  This way you'll be able
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

### 17.3.0 (unreleased)

* first published version

* date-versioned


## Authors

Glued together by [Łukasz Langa](mailto:lukasz@langa.pl).
