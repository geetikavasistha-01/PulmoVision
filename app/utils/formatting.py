"""
PulmoVision — UI Formatting Utilities
Provides consistent formatting for percentages, probabilities, latencies, and parameter counts.
"""

from typing import Optional, Union


def format_probability(prob: float) -> str:
    """Formats float probability into percentage string with 2 decimal places."""
    return f"{prob * 100.0:.2f}%"


def format_confidence_interval(ci_range: list) -> str:
    """Formats 2-element list into bracketed CI notation: [0.8984, 0.9820]."""
    if len(ci_range) == 2:
        return f"[{ci_range[0]:.4f}, {ci_range[1]:.4f}]"
    return str(ci_range)


def format_latency(ms: float) -> str:
    """Formats latency in milliseconds."""
    return f"{ms:.1f} ms"


def format_param_count(count: int) -> str:
    """Formats parameter count into human-readable string: e.g. 1,460,418 (~1.46M)."""
    if count >= 1_000_000:
        return f"{count:,} (~{count / 1_000_000:.2f}M)"
    if count >= 1_000:
        return f"{count:,} (~{count / 1_000:.1f}K)"
    return f"{count:,}"

