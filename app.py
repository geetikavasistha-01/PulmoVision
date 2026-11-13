"""
PulmoVision — Root Entrypoint Shim (Hugging Face Spaces & Streamlit Cloud).
Delegates execution directly to canonical app at app/streamlit_app.py without duplicating logic.
"""
import sys
from pathlib import Path
import runpy

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_APP = REPO_ROOT / "app" / "streamlit_app.py"

if not CANONICAL_APP.is_file():
    raise FileNotFoundError(f"Canonical entrypoint not found at {CANONICAL_APP}")

runpy.run_path(str(CANONICAL_APP), run_name="__main__")
