import os
import sys

# Get the current directory of the test file
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../src"))


def test_test_import():
    try:
        import test
    except ImportError:
        assert False, "Import of module 'test' failed."
