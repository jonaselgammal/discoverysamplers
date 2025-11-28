"""
Common prior parsing and model adapter utilities for Discovery sampler interfaces.

This module provides shared functionality for parsing prior specifications and
adapting Discovery models to various sampler backends.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import re


# Type definitions
ParamName = str
PriorSpec = Union[
    Mapping[str, Any],
    Tuple[str, float, float],
    Tuple[str, float, float, float],
    Tuple[str, float],
    Callable,
    float,
]


class PriorParsingError(ValueError):
    """Raised when a prior specification cannot be parsed."""
    pass


@dataclass
class ParsedPrior:
    """
    Unified representation of a parsed prior distribution.

    Attributes
    ----------
    dist_type : str
        Type of distribution: 'uniform', 'loguniform', 'normal', 'fixed', or 'callable'
    bounds : tuple of float, optional
        (min, max) bounds for the parameter
    mean : float, optional
        Mean for normal distribution
    sigma : float, optional
        Standard deviation for normal distribution
    value : float, optional
        Fixed value for 'fixed' distribution
    logpdf : callable, optional
        Log probability density function for callable priors
    """
    dist_type: str
    bounds: Optional[Tuple[float, float]] = None
    mean: Optional[float] = None
    sigma: Optional[float] = None
    value: Optional[float] = None
    logpdf: Optional[Callable[[float], float]] = None


def standard_priors(param_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """
    Build a prior dictionary using Discovery's ``priordict_standard`` patterns.

    Parameters
    ----------
    param_names : sequence of str
        Parameter names to match against Discovery's standard prior regexes.

    Returns
    -------
    dict
        Mapping ``name -> {'dist': 'uniform', 'min': a, 'max': b}``.

    Raises
    ------
    ImportError
        If the ``discovery`` package is not installed.
    KeyError
        If no standard prior is found for a parameter name.
    """
    try:
        import discovery.prior as dp
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError("Discovery is required to use standard_priors.") from exc

    priors: Dict[str, Dict[str, float]] = {}
    for name in param_names:
        matched = False
        for pattern, bounds in dp.priordict_standard.items():
            if re.match(pattern, name):
                priors[name] = {
                    "dist": "uniform",
                    "min": float(bounds[0]),
                    "max": float(bounds[1]),
                }
                matched = True
                break
        if not matched:
            raise KeyError(f"No standard prior found for parameter '{name}'.")
    return priors


def _parse_single_prior(name: str, spec: PriorSpec) -> ParsedPrior:
    """
    Parse a single prior specification into a ParsedPrior object.

    Parameters
    ----------
    name : str
        Parameter name
    spec : PriorSpec
        Prior specification in various formats:
        - Tuple: ('uniform', min, max), ('loguniform', min, max),
                 ('normal', mean, sigma), ('fixed', value)
        - Dict: {'dist': 'uniform', 'min': ..., 'max': ...}, etc.
        - Callable: Object with logpdf(value) method
        - Float: Interpreted as fixed value

    Returns
    -------
    ParsedPrior
        Parsed prior object

    Raises
    ------
    PriorParsingError
        If the specification cannot be parsed
    """
    # Handle callable priors
    if callable(spec):
        if not hasattr(spec, 'logpdf'):
            raise PriorParsingError(
                f"Callable prior for '{name}' must have a 'logpdf' method"
            )

        bounds = getattr(spec, 'bounds', (-np.inf, np.inf))
        return ParsedPrior(
            dist_type='callable',
            bounds=bounds,
            logpdf=spec.logpdf
        )

    # Handle scalar (fixed value)
    if isinstance(spec, (int, float)):
        return ParsedPrior(
            dist_type='fixed',
            value=float(spec)
        )

    # Handle tuple format
    if isinstance(spec, tuple):
        if len(spec) < 2:
            raise PriorParsingError(
                f"Tuple prior for '{name}' must have at least 2 elements"
            )

        dist_type = spec[0].lower()

        if dist_type == 'uniform':
            if len(spec) != 3:
                raise PriorParsingError(
                    f"Uniform prior for '{name}' requires (dist, min, max)"
                )
            return ParsedPrior(
                dist_type='uniform',
                bounds=(float(spec[1]), float(spec[2]))
            )

        elif dist_type == 'loguniform':
            if len(spec) != 3:
                raise PriorParsingError(
                    f"Log-uniform prior for '{name}' requires (dist, min, max)"
                )
            return ParsedPrior(
                dist_type='loguniform',
                bounds=(float(spec[1]), float(spec[2]))
            )

        elif dist_type == 'normal' or dist_type == 'gaussian':
            if len(spec) == 3:
                # (dist, mean, sigma)
                return ParsedPrior(
                    dist_type='normal',
                    mean=float(spec[1]),
                    sigma=float(spec[2])
                )
            elif len(spec) == 5:
                # (dist, mean, sigma, min, max) - truncated normal
                return ParsedPrior(
                    dist_type='normal',
                    mean=float(spec[1]),
                    sigma=float(spec[2]),
                    bounds=(float(spec[3]), float(spec[4]))
                )
            else:
                raise PriorParsingError(
                    f"Normal prior for '{name}' requires (dist, mean, sigma) "
                    f"or (dist, mean, sigma, min, max)"
                )

        elif dist_type == 'fixed':
            if len(spec) != 2:
                raise PriorParsingError(
                    f"Fixed prior for '{name}' requires (dist, value)"
                )
            return ParsedPrior(
                dist_type='fixed',
                value=float(spec[1])
            )

        else:
            raise PriorParsingError(
                f"Unknown prior type '{dist_type}' for parameter '{name}'"
            )

    # Handle dict format
    if isinstance(spec, dict):
        dist_type = spec.get('dist', '').lower()

        if dist_type == 'uniform':
            return ParsedPrior(
                dist_type='uniform',
                bounds=(float(spec['min']), float(spec['max']))
            )

        elif dist_type == 'loguniform':
            return ParsedPrior(
                dist_type='loguniform',
                bounds=(float(spec['min']), float(spec['max']))
            )

        elif dist_type == 'normal' or dist_type == 'gaussian':
            prior = ParsedPrior(
                dist_type='normal',
                mean=float(spec.get('mean', 0.0)),
                sigma=float(spec.get('sigma', spec.get('std', 1.0)))
            )
            # Optional bounds for truncated normal
            if 'min' in spec and 'max' in spec:
                prior.bounds = (float(spec['min']), float(spec['max']))
            return prior

        elif dist_type == 'fixed':
            return ParsedPrior(
                dist_type='fixed',
                value=float(spec['value'])
            )

        else:
            raise PriorParsingError(
                f"Unknown prior type '{dist_type}' for parameter '{name}'. "
                f"Supported types: uniform, loguniform, normal, fixed"
            )

    raise PriorParsingError(
        f"Cannot parse prior specification for '{name}': {spec}"
    )


def _split_priors(
    priors: Mapping[ParamName, PriorSpec]
) -> Tuple[
    List[str],  # sampled_names
    Dict[str, float],  # fixed_params
    Dict[str, Tuple[float, float]],  # bounds
    Dict[str, Callable[[float], float]]  # logprior_fns
]:
    """
    Split prior specifications into sampled and fixed parameters.

    This function parses all prior specifications and separates them into:
    - Sampled parameters with their bounds and log-prior functions
    - Fixed parameters with their values

    Parameters
    ----------
    priors : dict
        Dictionary mapping parameter names to prior specifications

    Returns
    -------
    sampled_names : list of str
        Names of parameters to be sampled
    fixed_params : dict
        Dictionary of fixed parameter values
    bounds : dict
        Dictionary of (min, max) bounds for sampled parameters
    logprior_fns : dict
        Dictionary of log-prior probability density functions

    Raises
    ------
    PriorParsingError
        If any prior specification cannot be parsed
    """
    sampled_names = []
    fixed_params = {}
    bounds = {}
    logprior_fns = {}

    for name, spec in priors.items():
        parsed = _parse_single_prior(name, spec)

        if parsed.dist_type == 'fixed':
            # Fixed parameter
            fixed_params[name] = parsed.value

        else:
            # Sampled parameter
            sampled_names.append(name)

            # Set bounds
            if parsed.bounds is None:
                # Unbounded (for normal distribution without truncation)
                bounds[name] = (-np.inf, np.inf)
            else:
                bounds[name] = parsed.bounds

            # Create log-prior function
            if parsed.dist_type == 'uniform':
                a, b = parsed.bounds
                log_width = np.log(b - a)

                def logprior_uniform(x, a=a, b=b, log_width=log_width):
                    if a <= x <= b:
                        return -log_width
                    return -np.inf

                logprior_fns[name] = logprior_uniform

            elif parsed.dist_type == 'loguniform':
                a, b = parsed.bounds
                log_a, log_b = np.log(a), np.log(b)
                norm = np.log(log_b - log_a)

                def logprior_loguniform(x, a=a, b=b, log_a=log_a, norm=norm):
                    if a <= x <= b:
                        return -np.log(x) - norm
                    return -np.inf

                logprior_fns[name] = logprior_loguniform

            elif parsed.dist_type == 'normal':
                mean = parsed.mean
                sigma = parsed.sigma
                log_norm = 0.5 * np.log(2 * np.pi * sigma**2)

                if parsed.bounds is None:
                    # Unbounded normal
                    def logprior_normal(x, mean=mean, sigma=sigma, log_norm=log_norm):
                        return -0.5 * ((x - mean) / sigma)**2 - log_norm

                    logprior_fns[name] = logprior_normal
                else:
                    # Truncated normal
                    a, b = parsed.bounds

                    def logprior_truncnormal(x, mean=mean, sigma=sigma, a=a, b=b, log_norm=log_norm):
                        if a <= x <= b:
                            return -0.5 * ((x - mean) / sigma)**2 - log_norm
                        return -np.inf

                    logprior_fns[name] = logprior_truncnormal

            elif parsed.dist_type == 'callable':
                # Use the provided logpdf
                logpdf = parsed.logpdf
                bounds_tuple = parsed.bounds

                def logprior_callable(x, logpdf=logpdf, bounds=bounds_tuple):
                    if bounds[0] <= x <= bounds[1]:
                        return logpdf(x)
                    return -np.inf

                logprior_fns[name] = logprior_callable

            else:
                raise PriorParsingError(
                    f"Unknown distribution type '{parsed.dist_type}' for '{name}'"
                )

    return sampled_names, fixed_params, bounds, logprior_fns
