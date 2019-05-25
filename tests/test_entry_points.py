import subprocess
import sys
from pathlib import Path


def test_has_module_access():
    subprocess.check_call([sys.executable, "-m", "retype", "--help"])


def test_has_script_call():
    subprocess.check_call([str(Path(sys.executable).parent / "retype"), "--help"])
