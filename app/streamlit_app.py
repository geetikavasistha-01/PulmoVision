"""PulmoVision — AI Medical Imaging Research Platform.

Main Streamlit Application Router.
Integrates Scientific AI Research Laboratory and Clinical Imaging Workstation.
"""

import sys
from pathlib import Path

# Ensure repository root is permanently at head of sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.styles.theme import apply_theme
from app.components.navigation import render_sidebar
from app.views.overview_view import render_overview_view
from app.views.dataset_view import render_dataset_view
from app.views.models_view import render_models_view
from app.views.analytics_view import render_analytics_view
from app.views.workstation_view import render_workstation_view
from app.views.explainability_view import render_explainability_view
from app.views.documentation_view import render_documentation_view

# 1. Page Configuration
st.set_page_config(
    page_title="PulmoVision — CT Sequence Intelligence & Research Platform",
    page_icon=":material/medical_services:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Sidebar Navigation & Dynamic Theme
destination, active_theme = render_sidebar()
apply_theme(active_theme)

# 3. Router Dispatch
if destination == "Research Overview":
    render_overview_view()
elif destination == "Dataset Explorer":
    render_dataset_view()
elif destination == "Model Laboratory":
    render_models_view()
elif destination == "Experiment Analytics":
    render_analytics_view()
elif destination == "CT Workstation":
    render_workstation_view()
elif destination == "Explainability":
    render_explainability_view()
elif destination == "Research Documentation":
    render_documentation_view()
else:
    st.error(f"Unknown destination: {destination}")
