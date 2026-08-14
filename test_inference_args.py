import subprocess
import sys


def test_config_path_required():
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f"Expected argparse to exit with code 2, got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
