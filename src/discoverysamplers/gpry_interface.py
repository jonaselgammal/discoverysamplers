"""
Discovery ↔︎ GPry bridge (via **Cobaya**)

This version is tailored to **GPry** and Cobaya's model wrapper.
It builds a Cobaya `model` (with your likelihood + priors) and passes that
`model` directly to `gpry.Runner`.

Key references:
- GPry Runner accepts a Cobaya `model` as its first argument; in that case it
  reads priors and names from the model (no `bounds` needed).
- In Cobaya, an external likelihood can be a plain Python function placed under
  the `likelihood` block, and priors are defined in the `params` block using
  scipy.stats names (e.g. `uniform`, `norm`, `loguniform`) or `min`/`max`.

What you get
------------
- `build_cobaya_info(...)` → a ready-to-use Cobaya `info` dict (likelihood + params)
- `get_cobaya_model(info)` → a `cobaya.model.Model`
- `DiscoveryGPryCobayaBridge` → convenience class that:
  1) adapts your Discovery-style likelihood
  2) builds the Cobaya model with proper priors
  3) launches `gpry.Runner(model, ...)` and exposes results

Assumptions
-----------
- Your *Discovery* model is either a callable `loglike(params_dict) -> float`
  or an object with `.log_prob(params_dict)` / `.log_likelihood(params_dict)`.
- You provide priors in a compact mapping per parameter.
  Supported entries here map to Cobaya priors:
    * ("uniform", a, b)  →  `prior: {min: a, max: b}`
    * ("loguniform", a, b)  →  `prior: {dist: loguniform, a: a, b: b}`
    * ("normal", mean, sigma[, ...])  →  `prior: {dist: norm, loc: mean, scale: sigma}`
    * ("fixed", value)  →  `value: value` (non-sampled)
  (These match Cobaya's documented syntax and SciPy parameterization. citeturn6search0)

If you need custom/callable priors, you can add them under Cobaya's top-level
`prior:` block—see the TODO note in `build_cobaya_info`.

Notes on parameter naming
-------------------------
Cobaya requires parameter names to be valid Python identifiers (no special chars
like `+`, `-`, etc.). This module automatically sanitizes parameter names by 
replacing invalid characters with underscores, and maintains a mapping to convert
back to the original names when calling the Discovery likelihood.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, List

import numpy as np

from .priors import ParsedPrior, PriorParsingError, _parse_single_prior, ParamName, PriorSpec


# ------------------------- Name sanitization ---------------------------- #

def _sanitize_param_name(name: str) -> str:
    """
    Convert a parameter name to a valid Python identifier for Cobaya.
    
    Replaces any character that's not alphanumeric or underscore with underscore.
    Ensures the name starts with a letter or underscore.
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized


def _create_name_mappings(param_names: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Create bidirectional mappings between original and sanitized parameter names.
    
    Returns
    -------
    orig_to_sanitized : dict
        Maps original names to sanitized names
    sanitized_to_orig : dict  
        Maps sanitized names back to original names
    """
    orig_to_sanitized = {}
    sanitized_to_orig = {}
    
    for name in param_names:
        sanitized = _sanitize_param_name(name)
        # Handle potential collisions by adding suffix
        base_sanitized = sanitized
        counter = 1
        while sanitized in sanitized_to_orig and sanitized_to_orig[sanitized] != name:
            sanitized = f"{base_sanitized}_{counter}"
            counter += 1
        orig_to_sanitized[name] = sanitized
        sanitized_to_orig[sanitized] = name
    
    return orig_to_sanitized, sanitized_to_orig


# --------------------------- GPry-specific adapter ---------------------- #

class _GPryDiscoveryAdapter:
    """
    Simplified Discovery adapter for GPry/Cobaya interface.

    This is a lightweight wrapper that doesn't use JAX JIT or array APIs,
    suitable for the Cobaya integration.
    """
    def __init__(self, model: Any):
        self.model = model
        if callable(model):
            self._fn = model
        elif hasattr(model, "log_prob") and callable(getattr(model, "log_prob")):
            self._fn = getattr(model, "log_prob")
        elif hasattr(model, "logL") and callable(getattr(model, "logL")):
            self._fn = getattr(model, "logL")
        else:
            raise TypeError(
                "Model must be callable or expose .log_prob/.logL(params_dict)."
            )

    def __call__(self, params: Mapping[str, float]) -> float:
        return float(self._fn(params))


# ------------------------------ Priors ---------------------------------- #

def split_priors(priors: Mapping[str, Any]) -> Tuple[List[str], Dict[str, float], Dict[str, ParsedPrior]]:
    """
    Split priors into sampled, fixed parameters, and parsed prior objects.

    Uses the common _parse_single_prior function but returns a format
    suitable for GPry/Cobaya.

    Returns
    -------
    sampled : list of str
        Names of sampled parameters
    fixed : dict
        Fixed parameter values
    parsed : dict
        ParsedPrior objects for each parameter
    """
    sampled: List[str] = []
    fixed: Dict[str, float] = {}
    parsed: Dict[str, ParsedPrior] = {}

    for name, spec in priors.items():
        p = _parse_single_prior(name, spec)
        parsed[name] = p
        if p.dist_type == "fixed":
            fixed[name] = p.value
        else:
            sampled.append(name)

    return sampled, fixed, parsed


# --------------------------- Cobaya builders ---------------------------- #

def _make_likelihood_func(parsed_prior: Mapping[str, ParsedPrior], adapter: _GPryDiscoveryAdapter,
                          sanitized_to_orig: Mapping[str, str],
                          fixed_params: Mapping[str, float]) -> Tuple[Callable[..., float], Tuple[str, ...]]:
    """Return a Cobaya-friendly likelihood.

    Cobaya accepts external likelihood *functions*; it will pass the sampled
    parameters by name as kwargs. We accept **kwargs to be robust to future
    changes in the parameterization. (Cobaya can introspect function args, but
    since we declare params explicitly in the `params` block, **kwargs is fine.)
    See Cobaya's advanced example for function likelihoods. citeturn5view0
    
    Parameters
    ----------
    parsed_prior : dict
        Mapping from original parameter names to ParsedPrior objects
    adapter : _GPryDiscoveryAdapter
        The likelihood adapter that accepts original parameter names
    sanitized_to_orig : dict
        Mapping from sanitized names (used by Cobaya) back to original names
    fixed_params : dict
        Mapping from original parameter names to fixed values
    
    Returns
    -------
    loglike : callable
        Likelihood function accepting sanitized parameter names as kwargs
    sampled_names : tuple
        Tuple of sanitized parameter names for sampled (non-fixed) parameters
    """
    # Get sanitized names for sampled parameters only
    orig_to_sanitized = {v: k for k, v in sanitized_to_orig.items()}
    sampled_sanitized = tuple(
        orig_to_sanitized[name] for name, p in parsed_prior.items() 
        if p.dist_type != "fixed"
    )
    
    # Store fixed params to include in every likelihood call
    _fixed_params = dict(fixed_params)

    def loglike(**kwargs: float) -> float:
        # Start with fixed parameters
        pd: Dict[str, float] = dict(_fixed_params)
        # Add sampled parameters (convert sanitized names back to original)
        for sanitized_name, value in kwargs.items():
            if sanitized_name in sanitized_to_orig:
                orig_name = sanitized_to_orig[sanitized_name]
                pd[orig_name] = float(value)
        return adapter(pd)

    return loglike, sampled_sanitized


def _prior_entry_from_parsed(p: ParsedPrior) -> Dict[str, Any] | float:
    """
    Convert a ParsedPrior from the common module to Cobaya format.

    Parameters
    ----------
    p : ParsedPrior
        Parsed prior from common module

    Returns
    -------
    dict or float
        Cobaya prior specification
    """
    if p.dist_type == "uniform":
        a, b = p.bounds
        return {"prior": {"min": a, "max": b}}
    if p.dist_type == "loguniform":
        a, b = p.bounds
        # Cobaya delegates to SciPy: loguniform takes 'a' (min) and 'b' (max).
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html
        return {"prior": {"dist": "loguniform", "a": a, "b": b}}
    if p.dist_type == "normal":
        mu, sigma = p.mean, p.sigma
        return {"prior": {"dist": "norm", "loc": mu, "scale": sigma}}
    if p.dist_type == "fixed":
        return p.value  # fixed parameter
    raise ValueError(f"Unsupported prior type for GPry/Cobaya: {p.dist_type}")


def build_cobaya_info(
    *,
    discovery_model: Any,
    priors: Mapping[str, Any],
    latex_labels: Optional[Mapping[str, str]] = None,
    like_name: str = "discovery_like",
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], Dict[str, float]]:
    """Create a Cobaya `info` dict with one external likelihood and a `params` block.

    Parameters
    ----------
    discovery_model : Any
        Discovery model (callable or object with logL method)
    priors : dict
        Prior specifications keyed by original parameter names
    latex_labels : dict, optional
        LaTeX labels keyed by original parameter names
    like_name : str
        Name for the likelihood in the Cobaya info

    Returns
    -------
    info : dict
        Cobaya info dict ready for get_model()
    orig_to_sanitized : dict
        Mapping from original parameter names to sanitized names
    sanitized_to_orig : dict
        Mapping from sanitized names back to original names
    fixed_params : dict
        Fixed parameter values (original names)

    Notes
    -----
    - Parameter names are automatically sanitized to be valid Python identifiers
    - Fixed parameters are baked into the likelihood function and NOT included
      in the Cobaya params block (to avoid "unused parameter" errors)
    - If you need external/callable *global* priors (beyond 1D priors in params),
      add them to `info['prior']` after this returns (see Cobaya docs). citeturn5view0
    """
    adapter = _GPryDiscoveryAdapter(discovery_model)
    sampled, fixed, parsed = split_priors(priors)
    
    # Create name mappings (original <-> sanitized)
    all_param_names = list(priors.keys())
    orig_to_sanitized, sanitized_to_orig = _create_name_mappings(all_param_names)

    # Likelihood function (external function style)
    # Pass fixed params so they are baked into the likelihood
    loglike, sampled_names = _make_likelihood_func(parsed, adapter, sanitized_to_orig, fixed)
    info: Dict[str, Any] = {"likelihood":
                            {like_name:
                             {"external": loglike,
                              "input_params": sampled_names}}}

    # Params block (using sanitized names, ONLY for sampled parameters)
    pblock: Dict[str, Any] = {}
    for orig_name, p in parsed.items():
        # Skip fixed parameters - they're baked into the likelihood
        if p.dist_type == "fixed":
            continue
            
        sanitized_name = orig_to_sanitized[orig_name]
        entry = _prior_entry_from_parsed(p)
        # add latex label if provided (use original name for lookup)
        if isinstance(entry, dict):
            if latex_labels and orig_name in latex_labels:
                entry = {**entry, "latex": latex_labels[orig_name]}
            elif orig_name != sanitized_name:
                # Use original name as latex label if no explicit label provided
                entry = {**entry, "latex": orig_name}
        pblock[sanitized_name] = entry

    info["params"] = pblock

    return info, orig_to_sanitized, sanitized_to_orig, fixed


def get_cobaya_model(info: Mapping[str, Any]):
    from cobaya.model import get_model
    return get_model(info)


# ------------------------------ GPry glue -------------------------------- #

class DiscoveryGPryCobayaBridge:
    """Bridge that:
    - builds a Cobaya model from a Discovery-style likelihood + priors
    - runs GPry by passing the Cobaya model to `gpry.Runner`
    
    Parameter names are automatically sanitized to be valid Python identifiers
    (required by Cobaya). The original names are stored and can be used for
    accessing results.
    
    Attributes
    ----------
    orig_param_names : list
        Original parameter names from the priors dict
    sanitized_param_names : list
        Sanitized parameter names used internally by Cobaya
    orig_to_sanitized : dict
        Mapping from original to sanitized names
    sanitized_to_orig : dict
        Mapping from sanitized to original names
    """

    def __init__(
        self,
        discovery_model: Any,
        priors: Mapping[str, Any],
        *,
        latex_labels: Optional[Mapping[str, str]] = None,
        like_name: str = "discovery_like",
        runner_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Store original parameter names
        self.orig_param_names = list(priors.keys())
        
        # Build Cobaya info with sanitized names
        self.info, self.orig_to_sanitized, self.sanitized_to_orig, self.fixed_param_dict = build_cobaya_info(
            discovery_model=discovery_model,
            priors=priors,
            latex_labels=latex_labels,
            like_name=like_name,
        )
        
        # Store sanitized names (only sampled params are in info["params"])
        self.sanitized_param_names = list(self.info["params"].keys())
        self.sampled_names_sanitized = list(self.info["params"].keys())
        self.fixed_names_sanitized = [self.orig_to_sanitized[n] for n in self.fixed_param_dict.keys()]
        
        # Also provide original names for convenience
        self.sampled_names = [self.sanitized_to_orig[n] for n in self.sampled_names_sanitized]
        self.fixed_names = list(self.fixed_param_dict.keys())
        
        self.model = get_cobaya_model(self.info)
        self.runner_kwargs = dict(runner_kwargs or {})
        self.runner = None
        self.results = None

    def run_sampler(self, *, checkpoint: Optional[str] = None, **run_kwargs: Any):
        """Create and run `gpry.Runner` with the Cobaya model.

        Per GPry docs, when passing a Cobaya `model` as first argument, GPry
        uses the model's prior and parameter names automatically. citeturn2view0
        
        Returns
        -------
        info : dict
            The Cobaya info dict used to create the model
        sampler : gpry.Runner
            The GPry Runner object after running
        """
        from gpry import Runner

        rkwargs = dict(self.runner_kwargs)
        if checkpoint is not None:
            rkwargs.setdefault("checkpoint", checkpoint)
        runner = Runner(self.model, **rkwargs)
        self.runner = runner
        self.results = runner.run(**run_kwargs)
        return self.info, runner

    # Convenience accessors (these use GPry's common post-run attributes)
    def posterior_samples(self) -> Optional[np.ndarray]:
        if self.runner is None:
            return None
        # GPry Cobaya wrapper stores samples via Cobaya products; but the Runner
        # can also expose `surrogate_samples` after MC on the GP. Try common names.
        for attr in ("posterior_samples", "surrogate_samples", "samples"):
            if hasattr(self.runner, attr):
                return np.asarray(getattr(self.runner, attr))
        return None

    def return_logZ(self, *, results=None) -> Dict[str, float]:
        """
        Return the log evidence estimate.
        
        Note: GPry uses Gaussian Process surrogate modeling and does not directly 
        compute the Bayesian evidence in the same way nested samplers do.
        This method is provided for API consistency but raises NotImplementedError.
        
        Raises
        ------
        NotImplementedError
            Always raised - GPry does not compute evidence in the standard nested sampling sense
        """
        raise NotImplementedError(
            "GPry uses Gaussian Process surrogate modeling and does not compute the "
            "Bayesian evidence (logZ) in the standard nested sampling sense. "
            "Use a nested sampling method (Nessai, JAX-NS) if you need evidence estimates."
        )

    def return_sampled_samples(self, *, results=None) -> Dict[str, Any]:
        """
        Return the sampled parameter chains.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'names': list of original parameter names
            - 'labels': list of parameter labels
            - 'chain': ndarray of shape (nsamples, n_sampled_params)
            
        Raises
        ------
        RuntimeError
            If no results are available
        """
        samples = self.posterior_samples()
        if samples is None:
            raise RuntimeError("No posterior samples available. Run `run_sampler()` first.")
        
        # GPry returns samples with sanitized names - we map back to original
        return {
            "names": self.sampled_names,
            "labels": self.sampled_names,  # Use original names as labels
            "chain": samples
        }

    def return_all_samples(self, *, results=None) -> Dict[str, Any]:
        """
        Return all samples including fixed parameters.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'names': list of all parameter names
            - 'labels': list of parameter labels  
            - 'chain': ndarray of shape (nsamples, n_all_params)
            
        Raises
        ------
        RuntimeError
            If no results are available
        """
        samples = self.posterior_samples()
        if samples is None:
            raise RuntimeError("No posterior samples available. Run `run_sampler()` first.")
        
        nsamp = samples.shape[0]
        chain_all = np.zeros((nsamp, len(self.orig_param_names)), dtype=float)
        
        # Fill sampled parameters
        for j, name in enumerate(self.sampled_names):
            idx = self.orig_param_names.index(name)
            chain_all[:, idx] = samples[:, j]
        
        # Fill fixed parameters
        for name in self.fixed_names:
            idx = self.orig_param_names.index(name)
            chain_all[:, idx] = float(self.fixed_param_dict[name])
        
        return {
            "names": self.orig_param_names,
            "labels": self.orig_param_names,
            "chain": chain_all
        }

    # ------------------------------ Plots ------------------------------ #
    def plot_trace(self, *, burn: int = 0, plot_fixed: bool = False, **kwargs):
        """
        Plot trace of samples vs sample index.
        
        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        plot_fixed : bool, optional
            If True, includes fixed parameters in the plot, by default False.
        **kwargs
            Additional keyword arguments passed to plots.plot_trace().
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the trace plots.
        """
        from .plots import plot_trace

        data = self.return_all_samples() if plot_fixed else self.return_sampled_samples()
        return plot_trace(
            data, 
            burn=burn,
            fixed_params=self.fixed_param_dict,
            fixed_names=self.fixed_names,
            **kwargs
        )

    def plot_corner(self, *, burn: int = 0, **kwargs):
        """
        Corner plot of sampled parameters.
        
        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        **kwargs
            Additional keyword arguments passed to corner.corner().
            
        Returns
        -------
        matplotlib.figure.Figure
            Corner plot figure.
        """
        from .plots import plot_corner

        data = self.return_sampled_samples()
        return plot_corner(data, burn=burn, **kwargs)


__all__ = [
    "build_cobaya_info",
    "get_cobaya_model",
    "DiscoveryGPryCobayaBridge",
]
