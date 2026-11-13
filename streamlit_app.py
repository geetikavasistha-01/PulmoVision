"""
PulmoVision — Streamlit Community Cloud Root Entrypoint Shim.
Directly invokes canonical entrypoint at app/streamlit_app.py without duplicating logic.
"""
import sys
from pathlib import Path
import runpy

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_APP = REPO_ROOT / "app" / "streamlit_app.py"

if not CANONICAL_APP.is_file():
    raise FileNotFoundError(f"Canonical entrypoint not found at {CANONICAL_APP}")

# Execute canonical entrypoint in current namespace
runpy.run_path(str(CANONICAL_APP), run_name="__main__")
