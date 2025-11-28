"""
Discovery ↔︎ nessai Interface

This module provides a light wrapper that adapts a Discovery-style model
(a callable that returns a log-probability/log-likelihood given a
parameter dictionary) to the API expected by `nessai`'s `FlowSampler`.

Design goals
------------
- Keep input style similar to the (intended) Eryn interface: pass a
  `discovery_model` and a `prior` dictionary.
- Minimise hard-coding so this can be refactored later into a common
  abstract `SamplerInterface`.
- Clean separation between: parsing priors, converting
  (dict ↔︎ structured array), and running the sampler.

Notes on nessai model API
-------------------------
`nessai` expects a subclass of `nessai.model.Model` that defines:
- `names: list[str]` – parameter names
- `bounds: dict[str, tuple[float, float]]` – hard bounds used for sampling
- `log_prior(x) -> float` – log prior for a single live point `x`
- `log_likelihood(x) -> float` – log likelihood for a single live point

`x` is a *structured numpy array* with fields equal to `names`.

This wrapper constructs such a model on the fly, based on a
`prior`-specification dictionary.

Prior specification
-------------------
The `prior` dictionary maps parameter names to specs. Supported forms:

1) Distribution dicts:
   {
     'dist': 'uniform' | 'loguniform' | 'normal' | 'fixed',
     # parameters depend on the dist (see `_make_prior`)
   }

2) Shorthand tuples for common cases:
   - ('uniform', min, max)
   - ('loguniform', a, b)
   - ('normal', mean, sigma)
   - ('fixed', value)

3) A callable prior: any object with `logpdf(value)` and, for non-fixed
   parameters, hard bounds provided via `'bounds': (min, max)`.

Fixed parameters are separated out and always injected into the model
inputs before calling the Discovery model.

Example
-------
>>> bridge = DiscoveryNessaiBridge(
...     discovery_model=my_model,  # callable or object with log_prob/log_likelihood
...     prior={
...         'm1': {'dist': 'uniform', 'min': 5, 'max': 50},
...         'm2': ('loguniform', 1e-1, 10.0),
...         'z':  ('fixed', 0.2),
...     },
... )
>>> # Run the sampler
>>> results = bridge.run_sampler(nlive=1000, max_iterations=50_000, output='./out')

"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import math
import os

import numpy as np
import numpy.lib.recfunctions as rfn

from .priors import ParsedPrior, PriorParsingError, _parse_single_prior, _split_priors, ParamName, PriorSpec
from .likelihood import LikelihoodWrapper

try:
    # These imports are only needed when actually running the sampler
    import nessai
    from nessai.flowsampler import FlowSampler
    from nessai.model import Model as NessaiModel
    import jax.numpy as jnp
except Exception:  # pragma: no cover - allow import without nessai installed
    FlowSampler = None  # type: ignore
    NessaiModel = object  # type: ignore
    jnp = None  # type: ignore


# -------------------------- Utilities & Types --------------------------- #

def _as_batch_struct(x: np.ndarray) -> np.ndarray:
    """Ensure x is a 1D structured array (length N)."""
    if x.dtype.names is None:
        raise TypeError("Expected a structured numpy array with named fields.")
    # np.atleast_1d keeps scalar structured arrays as shape (1,)
    xb = np.atleast_1d(x)
    if xb.ndim != 1:
        # nessai expects a 1D batch of structured samples
        xb = xb.reshape(-1)
    return xb


# ------------------------ Nessai-specific prior utilities --------------- #

def _convert_parsed_prior_to_nessai(parsed: ParsedPrior, name: str) -> Tuple[Optional[Tuple[float, float]], Dict[str, float]]:
    """
    Convert a ParsedPrior from the common module to nessai-compatible format.

    Returns
    -------
    bounds : tuple or None
        (min, max) bounds for the parameter
    params : dict
        Parameter-specific values (min, max for uniform, a, b for loguniform, etc.)
    """
    if parsed.dist_type == 'uniform':
        return parsed.bounds, {"min": parsed.bounds[0], "max": parsed.bounds[1]}
    elif parsed.dist_type == 'loguniform':
        return parsed.bounds, {"a": parsed.bounds[0], "b": parsed.bounds[1]}
    elif parsed.dist_type == 'normal':
        return parsed.bounds, {"mean": parsed.mean, "sigma": parsed.sigma}
    elif parsed.dist_type == 'fixed':
        return None, {"value": parsed.value}
    elif parsed.dist_type == 'callable':
        return parsed.bounds, {}
    else:
        raise PriorParsingError(f"Unsupported prior type '{parsed.dist_type}' for '{name}'")


def _split_priors_nessai(prior: Mapping[ParamName, PriorSpec]):
    """
    Split priors into sampled and fixed parameters for nessai.

    This is a nessai-specific wrapper around the common _split_priors function.
    It handles nessai's requirement for explicit bounds on normal priors.

    Returns
    -------
    sampled_names : list[str]
    fixed : dict[str, float]
    bounds : dict[str, tuple[float, float]]
    logprior_fns : dict[str, Callable[[float], float]]
    """
    # Use the common splitting function
    sampled_names, fixed, bounds, logprior_fns = _split_priors(prior)

    # Nessai-specific: check that all sampled parameters have finite bounds
    for name in sampled_names:
        if bounds[name] == (-np.inf, np.inf):
            raise PriorParsingError(
                f"Nessai requires finite bounds for all parameters. "
                f"Parameter '{name}' has infinite bounds. "
                f"For normal priors, specify bounds explicitly."
            )

    return sampled_names, fixed, bounds, logprior_fns


# --------------------------- nessai Model ------------------------------- #

class DiscoveryNessaiModel(NessaiModel):
    """A nessai `Model` that wraps a Discovery-style model.

    Parameters
    ----------
    names : list[str]
        Names of *sampled* parameters (fixed parameters are injected internally).
    bounds : dict[str, tuple[float, float]]
        Sampling bounds for each sampled parameter.
    logprior_fns : dict[str, Callable[[float], float]]
        Per-parameter log-prior functions.
    fixed_params : dict[str, float]
        Parameters that are not sampled.
    discovery_adapter : LikelihoodWrapper
        Adapter to call the Discovery model with a parameter dict.
    """
    def __init__(self,
                 names: List[str],
                 bounds: Dict[str, Tuple[float, float]],
                 logprior_fns: Dict[str, Callable[[float], float]],
                 fixed_params: Dict[str, float],
                 discovery_adapter: LikelihoodWrapper):
        super().__init__()
        self.names = list(names)                      
        self._names_tuple = tuple(self.names)          # optional: faster internal iteration

        self.bounds = dict(bounds)
        self._logprior_fns = dict(logprior_fns)
        self._fixed = dict(fixed_params)
        self._adapter = discovery_adapter
        
        # Allow non-deterministic likelihoods (disable verification check)
        self.allow_multi_valued_likelihood = True

        # Keep a fixed column order for packed matrices
        self._all_names = self._names_tuple + tuple(self._fixed.keys())
        if hasattr(self._adapter, "configure_array_api"):
            self._adapter.configure_array_api(self._all_names)

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        xb = _as_batch_struct(x)                 # shape (N,)
        N = xb.shape[0]
        total = np.zeros(N, dtype=float)
        for n in self.names:
            fn = self._logprior_fns[n]
            vals = xb[n]                         # shape (N,)
            # Evaluate per-sample; robust if fn is scalar-only
            contrib = np.array([float(fn(float(v))) for v in vals], dtype=float)
            total += contrib

        # Replace any non-finite totals with -inf (per-sample)
        total[~np.isfinite(total)] = -np.inf

        # Preserve scalar-like return for N=1 while keeping ndarray type
        return total if N > 1 else np.array(total[0])

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        xb = _as_batch_struct(x)                 # shape (N,)
        N = xb.shape[0]
        # Build one dense host array (N, D_s)
        cols = [xb[n].astype(np.float64, copy=False) for n in self._names_tuple]
        X = np.column_stack(cols)  # (N, D_s)
        # Append fixed params on host
        if self._fixed:
            fvec = np.array([self._fixed[k] for k in self._fixed], dtype=np.float64)  # (D_f,)
            if fvec.size:
                F = np.broadcast_to(fvec, (N, fvec.size))                              # (N, D_f)
                X = np.concatenate([X, F], axis=1)                                    # (N, D)
        # One device put, zero dicts:
        Xj = jnp.asarray(X, dtype=jnp.float64)
        return self._adapter.log_likelihood_matrix(Xj)
    
    # nessai calls these with a *structured* numpy array `x`
    def _log_prior(self, x: np.ndarray) -> float:
        total = 0.0
        for n in self.names:
            total += float(self._logprior_fns[n](float(x[n])))
            if not np.isfinite(total):
                return -np.inf
        return np.array(total)

    def _log_likelihood(self, x: np.ndarray) -> float:
        # Scalar path: pack one row -> array -> compiled row call (no dicts here either)
        vals = [float(x[n]) for n in self._names_tuple]
        if self._fixed:
            vals.extend(float(self._fixed[k]) for k in self._fixed)
        row = jnp.asarray(np.asarray(vals, dtype=np.float64))
        return self._adapter.log_likelihood_row(row)


# ------------------------- Public bridge class ------------------------- #

class DiscoveryNessaiBridge:
    """Bridge between a Discovery-style model and `nessai`'s FlowSampler.

    Parameters
    ----------
    discovery_model : callable | object
        A callable or object with `log_prob` or `log_likelihood`.
    priors : Mapping[str, PriorSpec]
        Dictionary describing priors (see module docstring). Includes fixed
        parameters.
    labels : Optional[Mapping[str, str]]
        Optional display labels per parameter (not used internally yet).

    Attributes
    ----------
    sampled_names : list[str]
    fixed_params : dict[str, float]
    bounds : dict[str, tuple[float, float]]
    model : DiscoveryNessaiModel
    """

    def __init__(self,
                 discovery_model: Any,
                 priors: Mapping[str, PriorSpec],
                 latex_labels: Optional[Mapping[str, str]] = None,
                 jit: bool = True):
        self.adapter = LikelihoodWrapper(discovery_model, jit=jit, fixed_params=None, allow_array_api=True)
        snames, fixed, bounds, lpfns = _split_priors_nessai(priors)
        if not snames:
            raise ValueError("No sampled parameters defined (all fixed?)")
        self.sampled_names = snames
        self.fixed_params = fixed
        self.bounds = bounds
        self.latex_labels = dict(latex_labels) if latex_labels else {n: n for n in snames}

        # Keep original order from the priors dict for "all params"
        self.discovery_paramnames = list(priors.keys())
        self.fixed_names = list(self.fixed_params.keys())

        # Build label lists
        # self.latex_labels already exists in your class; ensure it's a mapping name->label
        # If a label is missing, fall back to the name.
        self.latex_list = [self.latex_labels.get(n, n) for n in self.discovery_paramnames]
        self.sampled_names_latex = [self.latex_labels.get(n, n) for n in self.sampled_names]
        self.fixed_names_latex = [self.latex_labels.get(n, n) for n in self.fixed_names]

        self.model = DiscoveryNessaiModel(
            names=self.sampled_names,
            bounds=self.bounds,
            logprior_fns=lpfns,
            fixed_params=self.fixed_params,
            discovery_adapter=self.adapter
        )

    # Convenience helpers -------------------------------------------------
    def dict_to_livepoint(self, d: Mapping[str, float]) -> np.ndarray:
        """Convert a parameter dict to a nessai live point (structured array)."""
        dt = [(n, "f8") for n in self.sampled_names] + [("logP", "f8"), ("logL", "f8")]
        lp = np.zeros((), dtype=dt)
        for n in self.sampled_names:
            lp[n] = float(d[n])
        lp["logP"] = 0.0
        lp["logL"] = 0.0
        return lp

    def livepoint_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        return {n: float(x[n]) for n in self.sampled_names}

    # Running the sampler -------------------------------------------------
    def run_sampler(self, *,
                    nlive: int = 1000,
                    output: str = "./nessai_out",
                    resume: bool = False,
                    **kwargs: Any) -> Any:
        """Run `nessai.FlowSampler` with this model.

        Parameters
        ----------
        nlive : int
            Number of live points.
        output : str
            Output directory for nessai (checkpoints, samples, plots).
        max_iterations : Optional[int]
            Maximum number of iterations. If `None`, uses nessai default.
        seed : Optional[int]
            Random seed passed to the sampler.
        resume : bool
            Resume from previous run if possible.
        **kwargs : Any
            Forwarded directly to `FlowSampler`.

        Returns
        -------
        The object returned by `FlowSampler.run()`, typically a results
        dictionary including posterior samples and evidences.
        """
        if FlowSampler is None:
            raise RuntimeError("nessai is not installed. Please `pip install nessai`." )

        self.sampler = FlowSampler(
            self.model,
            output=output,
            nlive=nlive,
            resume=resume,
            **kwargs,
        )
        self.results = self.sampler.run()
        
        # If run() returns None (newer nessai versions), construct results dict from sampler state
        if self.results is None and hasattr(self.sampler, 'ns') and hasattr(self.sampler.ns, 'state'):
            state = self.sampler.ns.state
            self.results = {
                'logZ': state.log_evidence if hasattr(state, 'log_evidence') else None,
                'logZ_err': state.log_evidence_error if hasattr(state, 'log_evidence_error') else None,
                'nested_samples': getattr(self.sampler.ns, 'nested_samples', None),
                'posterior_samples': getattr(self.sampler, 'posterior_samples', None),
            }

        return self.results
    
    def return_logZ(self, *, results: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
        """
        Return the log evidence and its uncertainty from nested sampling.
        
        Parameters
        ----------
        results : dict, optional
            Results dict from run_sampler(). If None, uses stored results.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'logZ': float - the log evidence estimate
            - 'logZ_err': float - uncertainty on logZ
            
        Raises
        ------
        RuntimeError
            If no results are available (run_sampler not called)
        """
        res = results if results is not None else self.results
        if res is None:
            raise RuntimeError("No results available. Run `run_sampler()` first.")
        
        # Try to extract logZ from results dict
        logZ = None
        logZ_err = None
        
        # Check common keys in results dict
        for key in ('logZ', 'log_evidence', 'log_Z', 'evidence'):
            if key in res:
                logZ = float(res[key])
                break
        
        for key in ('logZ_err', 'log_evidence_error', 'log_Z_err', 'evidence_error', 'logZerr'):
            if key in res:
                logZ_err = float(res[key])
                break
        
        # If not found in results, try sampler state
        if logZ is None and self.sampler is not None:
            if hasattr(self.sampler, 'ns') and hasattr(self.sampler.ns, 'state'):
                state = self.sampler.ns.state
                if hasattr(state, 'log_evidence'):
                    logZ = float(state.log_evidence)
                if hasattr(state, 'log_evidence_error'):
                    logZ_err = float(state.log_evidence_error)
        
        if logZ is None:
            raise RuntimeError("Could not find log evidence in results. Check that sampling completed successfully.")
        
        return {'logZ': logZ, 'logZ_err': logZ_err}

    # ------------------------ Results & samples ------------------------ #
    def _posterior_struct_array(self, results: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        """
        Return a *structured* numpy array of posterior samples (one row per sample)
        with fields that include all `self.sampled_names` (plus possibly logL/logP/weights).
        Tries several common locations and errors with guidance if not found.
        """
        # 1) explicitly provided results dict
        if results is not None:
            for k in ("posterior_samples", "samples", "posterior"):
                if k in results and isinstance(results[k], np.ndarray) and results[k].dtype.names:
                    return results[k]

        # 2) whatever run() returned last time
        if self.results is not None:
            for k in ("posterior_samples", "samples", "posterior"):
                v = self.results.get(k) if isinstance(self.results, Mapping) else None
                if isinstance(v, np.ndarray) and v.dtype.names:
                    return v

        # 3) look on the sampler object (common on recent nessai)
        if self.sampler is not None:
            for attr in ("posterior_samples", "posterior", "samples", "samples_posterior"):
                v = getattr(self.sampler, attr, None)
                if isinstance(v, np.ndarray) and v.dtype.names:
                    return v
            # 4) fall back to the output directory (best effort; optional)
            out = getattr(self.sampler, "output", None)
            if isinstance(out, str) and os.path.isdir(out):
                for fname in ("posterior_samples.npy", "posterior.npy", "samples_post.npy"):
                    path = os.path.join(out, fname)
                    if os.path.exists(path):
                        arr = np.load(path, allow_pickle=False)
                        if isinstance(arr, np.ndarray) and arr.dtype.names:
                            return arr

        raise RuntimeError(
            "Could not locate posterior samples. "
            "Run `run_sampler()` first, or pass the results dict returned by `.run_sampler()`."
        )

    def _stack_columns(self, struct: np.ndarray, names: Iterable[str]) -> np.ndarray:
        """Return a (nsamples, len(names)) float array by selecting fields from a structured array."""
        cols = [np.asarray(struct[n], dtype=float).reshape(-1) for n in names]
        return np.stack(cols, axis=1) if cols else np.empty((len(struct), 0), dtype=float)

    # --------------------------- Public API ---------------------------- #
    def return_sampled_samples(self, *, results: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns sampled parameters only.

        Returns
        -------
        dict with keys:
          - 'names' : list[str]
          - 'labels': list[str] (LaTeX or names)
          - 'chain' : ndarray (nsamples, n_sampled)
        """
        struct = self._posterior_struct_array(results)
        chain = self._stack_columns(struct, self.sampled_names)
        return {"names": self.sampled_names, "labels": self.sampled_names_latex, "chain": chain}

    def return_all_samples(self, *, results: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns sampled + fixed parameters, arranged in the original `priors` order.
        Fixed parameters are filled with their constant values.
        """
        struct = self._posterior_struct_array(results)
        ns = struct.shape[0]
        chain_all = np.zeros((ns, len(self.discovery_paramnames)), dtype=float)

        # Fill sampled
        sampled_cols = self._stack_columns(struct, self.sampled_names)
        for j, name in enumerate(self.sampled_names):
            idx = self.discovery_paramnames.index(name)
            chain_all[:, idx] = sampled_cols[:, j]

        # Fill fixed
        for name in self.fixed_names:
            idx = self.discovery_paramnames.index(name)
            chain_all[:, idx] = float(self.fixed_params[name])

        return {"names": self.discovery_paramnames, "labels": self.latex_list, "chain": chain_all}

    # ------------------------------ Plots ------------------------------ #
    def plot_trace(self, *, burn: int = 0, plot_fixed: bool = False,
                   results: Optional[Mapping[str, Any]] = None):
        """
        Plot simple trace(s) vs sample index (no walkers/temps in nessai).
        Returns a matplotlib.figure.Figure.
        """
        import matplotlib.pyplot as plt

        data = self.return_all_samples(results=results) if plot_fixed else self.return_sampled_samples(results=results)
        chain = data["chain"]
        names = data["names"]
        labels = data["labels"]

        if burn > 0:
            chain = chain[burn:]

        n_params = len(names)
        fig, axes = plt.subplots(n_params, 1, figsize=(9, max(2.2, 1.8 * n_params)), sharex=True)

        axes_arr = np.atleast_1d(axes)
        for i, name in enumerate(names):
            ax = axes_arr[i]
            ax.plot(chain[:, i], lw=0.7, alpha=0.9)
            ax.set_ylabel(labels[i])
            if plot_fixed and name in self.fixed_names:
                ax.axhline(float(self.fixed_params[name]), ls="--", lw=1.0, color="r", label="fixed")
                ax.legend(loc="best", frameon=False)
        axes_arr[-1].set_xlabel("sample index")
        fig.tight_layout()
        return fig

    def plot_corner(self, *, burn: int = 0, results: Optional[Mapping[str, Any]] = None, **kwargs):
        """
        Corner plot of sampled parameters (single temperature/chain).
        Returns a matplotlib.figure.Figure.
        """
        import corner

        data = self.return_sampled_samples(results=results)
        chain = data["chain"]
        if burn > 0:
            chain = chain[burn:]
        labels = data["labels"]
        fig = corner.corner(chain, labels=labels, **kwargs)
        return fig



__all__ = [
    "DiscoveryNessaiBridge",
    "DiscoveryNessaiModel",
]