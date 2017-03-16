# retype

[![Build Status](https://travis-ci.org/ambv/retype.svg?branch=master)](https://travis-ci.org/ambv/retype)

Re-apply type annotations from .pyi stubs to your codebase.


## This is a work in progress.

Things to be done:

* [x] reapply typing imports
* [x] reapply function argument annotations
* [x] reapply function return value annotations
* [x] reapply method argument and return value annotations
* [x] reapply function-level variable annotations
* [ ] reapply module-level field annotations
* [ ] reapply module-level type aliases
* [ ] reapply class-level field annotations
* [ ] reapply instance-level field annotations
* [ ] add a --python2 option and remove print_statement by default
* [ ] add the --keep-byte-literals option for Mercurial
* [ ] support type comments in .pyi files
* [ ] add a --backward option to output type comments instead of annotations
* [ ] handle if sys.version_info and sys.platform checks in stubs
* [ ] warn about functions and classes with missing annotations


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
