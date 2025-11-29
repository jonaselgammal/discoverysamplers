"""
discoverysamplers: Bridge interfaces for Discovery models to various samplers.

This package provides lightweight wrappers that adapt Discovery-style models
to the APIs expected by different sampling backends.
"""

__version__ = "0.1.0"

# Export common utilities
from .priors import (
    ParsedPrior,
    PriorParsingError,
    ParamName,
    PriorSpec,
    standard_priors,
)

# Export plotting functions
from .plots import (
    plot_trace,
    plot_corner,
    plot_corner_multi_temp,
    plot_nleaves_histogram,
    plot_parameter_summary,
    plot_run_plot,
)

__all__ = [
    # Priors
    "ParsedPrior",
    "PriorParsingError",
    "ParamName",
    "PriorSpec",
    "standard_priors",
    # Plotting
    "plot_trace",
    "plot_corner",
    "plot_corner_multi_temp",
    "plot_nleaves_histogram",
    "plot_parameter_summary",
    "plot_run_plot",
]
