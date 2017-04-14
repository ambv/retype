# Copyright (C) 2017 Łukasz Langa

import ast
import os
import re
from setuptools import setup
import sys

try:
    from pathlib import Path
except ImportError:
    pass  # will fail assert below.

assert sys.version_info >= (3, 6, 0), "retype requires Python 3.6+"


current_dir = os.path.abspath(os.path.dirname(__file__))
readme_md = os.path.join(current_dir, 'README.md')
try:
    import pypandoc
    long_description = pypandoc.convert_file(readme_md, 'rst')
except(IOError, ImportError):
    print()
    print(
        '\x1b[31m\x1b[1mwarning:\x1b[0m\x1b[31m pandoc not found, '
        'long description will be ugly (PyPI does not support .md).'
        '\x1b[0m'
    )
    print()
    with open(readme_md, encoding='utf8') as ld_file:
        long_description = ld_file.read()


_version_re = re.compile(r'__version__\s+=\s+(?P<version>.*)')


with open(os.path.join(current_dir, 'retype.py'), 'r', encoding='utf8') as f:
    version = _version_re.search(f.read()).group('version')
    version = str(ast.literal_eval(version))


setup(
    name='retype',
    version=version,
    description="Re-apply types from .pyi stub files to your codebase.",
    long_description=long_description,
    keywords='mypy typing typehints type hints pep484 pyi stubs',
    author='Łukasz Langa',
    author_email='lukasz@langa.pl',
    url='https://github.com/ambv/retype',
    license='MIT',
    py_modules=['retype', 'retype_hgext'],
    data_files=[
        (
            str(Path('lib/mypy/typeshed/third_party/3.6')),
            ('types/retype.pyi', 'types/retype_hgext.pyi'),
        ),
    ],
    zip_safe=False,
    install_requires=['click', 'typed-ast'],
    test_suite='tests.test_retype',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Quality Assurance',
    ],
    entry_points={
        'console_scripts': ['retype=retype:main'],
    },
)
