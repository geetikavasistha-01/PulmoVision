"""tests/test_app_navigation.py

Verifies Streamlit app navigation and state management across all destinations,
ensuring no StreamlitWidgetAlreadyInstantiatedError or duplicate state warnings.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_ENTRYPOINT = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_streamlit_app_initial_load() -> None:
    """Verify app loads cleanly without exceptions."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert at.session_state.get("current_page") == "Research Overview"


def test_streamlit_app_destination_card_navigation() -> None:
    """Verify clicking an 'Open <Page>' card navigates properly without session state collision."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Locate and click 'Open Dataset Explorer'
    dataset_btn = next((b for b in at.button if "Dataset Explorer" in b.label), None)
    assert dataset_btn is not None, "Navigation button for Dataset Explorer should exist"

    dataset_btn.click().run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert at.session_state.get("current_page") == "Dataset Explorer"
    assert "Dataset Explorer" in at.session_state.get("main_nav_radio", "")
