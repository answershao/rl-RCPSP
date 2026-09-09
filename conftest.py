"""Repo-root conftest: make ``src`` / ``tests`` importable under pytest.

Running ``pytest`` from the repository root is the supported workflow; this
file additionally keeps imports working when pytest is launched from a
sub-directory (e.g. ``pytest tests/``).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
