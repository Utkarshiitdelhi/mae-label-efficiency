import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: longer CPU tests (one-batch overfit)")
