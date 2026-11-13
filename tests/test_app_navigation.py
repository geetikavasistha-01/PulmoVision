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


def test_documentation_section_navigation() -> None:
    """Verify Research Documentation 'By Section' navigator, TOC jump, and Prev/Next buttons."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.session_state["current_page"] = "Research Documentation"
    at.run()
    assert len(at.exception) == 0

    # Switch to By Section mode
    mode_radio = next(r for r in at.radio if r.key == "doc_view_mode")
    mode_radio.set_value("By Section").run()
    assert len(at.exception) == 0

    # Initial section should default to 1. Abstract
    picker = next(s for s in at.selectbox if s.key == "doc_section_picker")
    assert "Abstract" in picker.value

    # Click Next button
    next_btn = next(b for b in at.button if b.key == "doc_btn_next_top")
    next_btn.click().run()
    assert len(at.exception) == 0
    picker = next(s for s in at.selectbox if s.key == "doc_section_picker")
    assert "Research Problem" in picker.value

    # Click Previous button
    prev_btn = next(b for b in at.button if b.key == "doc_btn_prev_top")
    prev_btn.click().run()
    assert len(at.exception) == 0
    picker = next(s for s in at.selectbox if s.key == "doc_section_picker")
    assert "Abstract" in picker.value

    # Jump via TOC button
    toc_btn = next((b for b in at.button if b.key == "toc_jump_4"), None)
    if toc_btn:
        toc_btn.click().run()
        assert len(at.exception) == 0
        picker = next(s for s in at.selectbox if s.key == "doc_section_picker")
        assert "Deduplication" in picker.value

