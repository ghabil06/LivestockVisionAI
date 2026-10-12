"""
Pytest configuration within tests/ directory.
Guarantees LivestockVisionAI is discoverable even when invoking `pytest tests/` directly.
"""

import sys
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
pkg_root = tests_dir.parent
parent_dir = pkg_root.parent

for p in [str(parent_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
