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

__all__ = [
    "ParsedPrior",
    "PriorParsingError",
    "ParamName",
    "PriorSpec",
    "standard_priors",
]
