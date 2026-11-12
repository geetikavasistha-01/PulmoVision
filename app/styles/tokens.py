"""
PulmoVision — Design System Tokens
Defines semantic color palettes, typography, spacing, and component styles for Dark and Light themes.
"""

from typing import Dict, Any


THEMES: Dict[str, Dict[str, Any]] = {
    "dark": {
        "name": "Clinical Dark",
        "bg_app": "#0a0e17",
        "bg_surface": "#131b26",
        "bg_elevated": "#1c2533",
        "bg_card": "#151d2a",
        "border": "#243042",
        "border_subtle": "#1b2533",
        "text_primary": "#f1f5f9",
        "text_secondary": "#94a3b8",
        "text_muted": "#64748b",
        "accent_primary": "#00b4d8",     # Muted cyan/teal
        "accent_secondary": "#38bdf8",   # Soft sky blue
        "accent_hover": "#0096c7",
        "success": "#10b981",            # Restrained green
        "warning": "#f59e0b",            # Muted amber
        "error": "#ef4444",              # Restrained red
        "info": "#0284c7",
        "workstation_bg": "#05070a",     # Pure deep dark canvas for CT inspection
        "workstation_frame": "#1e293b",
        "font_sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font_mono": "'JetBrains Mono', 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace",
        "shadow_card": "0 2px 8px rgba(0, 0, 0, 0.4)",
    },
    "light": {
        "name": "Laboratory Light",
        "bg_app": "#f8fafc",
        "bg_surface": "#ffffff",
        "bg_elevated": "#f1f5f9",
        "bg_card": "#ffffff",
        "border": "#e2e8f0",
        "border_subtle": "#edf2f7",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "text_muted": "#64748b",
        "accent_primary": "#0f766e",     # Deep professional teal
        "accent_secondary": "#0284c7",   # Deep blue
        "accent_hover": "#0d9488",
        "success": "#059669",
        "warning": "#d97706",
        "error": "#dc2626",
        "info": "#0369a1",
        "workstation_bg": "#0a0e17",     # Preserved dark viewing canvas for medical CT
        "workstation_frame": "#334155",
        "font_sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font_mono": "'JetBrains Mono', 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace",
        "shadow_card": "0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)",
    }
}


def get_theme(theme_name: str = "dark") -> Dict[str, Any]:
    """Returns theme token dictionary, defaulting to dark theme."""
    return THEMES.get(theme_name.lower(), THEMES["dark"])


# Shared Constants
CT_CANVAS_COLOR = "#000000"
FONT_FAMILY_BASE = THEMES["dark"]["font_sans"]
FONT_FAMILY_MONO = THEMES["dark"]["font_mono"]


