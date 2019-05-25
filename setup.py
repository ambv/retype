"""Handle the package management of the module"""
from setuptools import setup

setup(
    use_scm_version={
        "write_to": "src/retype/version.py",
        "write_to_template": '__version__ = "{version}"',
    }
)
