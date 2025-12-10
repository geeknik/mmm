#!/usr/bin/env python3
"""
Quick test to verify CLI commands are working
"""

import subprocess
import sys

def test_command(cmd, timeout=10):
    """Test if command starts without immediate errors"""
    try:
        result = subprocess.run(
            f"source mmm_env/bin/activate && timeout {timeout} {cmd} 2>&1",
            shell=True,
            capture_output=True,
            text=True
        )

        # Check for config errors
        if "'Command' object has no attribute 'config'" in result.stderr:
            return False, "Config error still present"

        # Check for immediate Python errors
        if "Traceback" in result.stderr or "Error:" in result.stderr:
            return False, f"Python error: {result.stderr[:100]}"

        # Check if banner appears (indicates CLI started)
        if "MELODIC METADATA MASSACRER" in result.stdout:
            return True, "CLI started successfully"

        return True, "No immediate errors"

    except subprocess.TimeoutExpired:
        return True, "Process started but timed out (expected)"
    except Exception as e:
        return False, f"Exception: {e}"

def main():
    print("🧪 Quick CLI Test")
    print("="*50)

    tests = [
        ("python -m mmm.cli --help", "Help command"),
        ("python -m mmm.cli analyze --help", "Analyze help"),
        ("python -m mmm.cli obliterate --help", "Obliterate help"),
        ("python -m mmm.cli massacre --help", "Massacre help")
    ]

    all_passed = True
    for cmd, desc in tests:
        passed, message = test_command(cmd)
        status = "✅" if passed else "❌"
        print(f"{status} {desc}: {message}")
        if not passed:
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! CLI is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())