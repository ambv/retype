import subprocess
import sys
from pathlib import Path


def test_has_module_access():
    subprocess.check_call([sys.executable, "-m", "retype", "--help"])


def test_has_script_call():
    subprocess.check_call([Path(sys.executable).parent / "retype", "--help"])


def test_directly_invoke_able():
    import retype

    cmd = [sys.executable, str(Path(retype.__file__).parent / "__init__.py"), "--help"]
    subprocess.check_call(cmd)
