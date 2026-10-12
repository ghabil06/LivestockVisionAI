"""
Pytest configuration for LivestockVisionAI.
Ensures universal package resolution regardless of invocation working directory.
"""

import sys
from pathlib import Path

# Add LivestockVisionAI and its parent directory to sys.path
root_dir = Path(__file__).resolve().parent
parent_dir = root_dir.parent

for p in [str(parent_dir), str(root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
