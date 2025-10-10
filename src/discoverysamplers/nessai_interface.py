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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import math

import numpy as np
import numpy.lib.recfunctions as rfn

try:
    # These imports are only needed when actually running the sampler
    import nessai
    from nessai.flowsampler import FlowSampler
    from nessai.model import Model as NessaiModel
except Exception:  # pragma: no cover - allow import without nessai installed
    FlowSampler = None  # type: ignore
    NessaiModel = object  # type: ignore

import jax.numpy as jnp


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

ParamName = str
PriorSpec = Union[
    Mapping[str, Any],              # dict specification
    Tuple[str, float, float],       # ('uniform', min, max) or ('loguniform', a, b)
    Tuple[str, float, float, float],# ('normal', mean, sigma, _unused_)
    Tuple[str, float],              # ('fixed', value)
]


@dataclass
class ParsedPrior:
    dist: str  # 'uniform' | 'loguniform' | 'normal' | 'fixed' | 'callable'
    bounds: Optional[Tuple[float, float]]
    params: Dict[str, float]
    # For callable priors, a callable with signature logpdf(x) -> float
    logpdf: Optional[Callable[[float], float]] = None


# ------------------------ Discovery model adapter ----------------------- #

class __DiscoveryAdapter:
    """Uniform interface to call a Discovery-style model.
    Parameters
    ----------
    model : callable | object
        A callable or object with `logL` method.
    jit : bool
        Whether to try to use JAX Just-In-Time compilation.
    """

    def _resolve_jit(self, fn: Callable, jit: bool) -> Callable:
        try:
            import jax
        except ImportError:
            raise ImportError("JAX not installed. You can't jit without jax.")
        if jit:
            try:
                return jax.jit(fn)
            except Exception as e:
                print(f"Warning: failed to jit-compile the model: {e}")
        return fn


    def __init__(self, model: Any, jit=True):
        self.model = model
        # Resolve once to avoid branches in the hot path
        if callable(model):
            self._fn = self._resolve_jit(model, jit) if jit else model
        elif hasattr(model, "logL") and callable(getattr(model, "logL")):
            self._fn = self._resolve_jit(model.logL, jit) if jit else model.logL
        else:
            raise TypeError(
                "`discovery_model` must be callable or have a `logL` method."
            )

    def log_likelihood(self, params: Mapping[str, float]) -> float:
        return self._fn(params)

class old_DiscoveryAdapter:
    """Uniform interface to call a Discovery-style model, with batch support."""
    def __init__(self, model: Any, jit: bool = True):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX not installed. You can't jit without jax.")

        # Resolve scalar function once
        if callable(model):
            fn = model
        elif hasattr(model, "logL") and callable(getattr(model, "logL")):
            fn = model.logL
        else:
            raise TypeError("`model` must be callable or have a `logL` method.")

        # Scalar path (single set of params)
        self._fn = jax.jit(fn) if jit else fn

        # Batched path: map over leading axis of each leaf in the params pytree
        # No static args; dict keys are static by structure.
        self._batched_fn = jax.jit(jax.vmap(self._fn)) if jit else jax.vmap(self._fn)

    def log_likelihood(self, params: Mapping[str, float]) -> float:
        # scalar params (dict of 0-d arrays or Python floats)
        return self._fn(params)

    def log_likelihood_batch(self, params: Mapping[str, "jnp.ndarray"]) -> "jnp.ndarray":
        # params is a dict of arrays with matching leading dimension
        return self._batched_fn(params)
    

class _DiscoveryAdapter:
    def __init__(self, model: Any, jit=True, order=None):
        self.model = model
        # Resolve once to avoid branches in the hot path
        if callable(model):
            self._fn = self._resolve_jit(model, jit) if jit else model
        elif hasattr(model, "logL") and callable(getattr(model, "logL")):
            self._fn = self._resolve_jit(model.logL, jit) if jit else model.logL
        else:
            raise TypeError(
                "`discovery_model` must be callable or have a `logL` method."
            )

    def _resolve_jit(self, fn: Callable, jit: bool) -> Callable:
        try:
            import jax
        except ImportError:
            raise ImportError("JAX not installed. You can't jit without jax.")
        if jit:
            try:
                return jax.jit(fn)
            except Exception as e:
                print(f"Warning: failed to jit-compile the model: {e}")
        return fn

    def log_likelihood(self, params: Mapping[str, float]) -> float:
        return self._fn(params)

    # -------- New: compiled array interfaces (no dicts in the hot path) --------
    def configure_array_api(self, order):
        """Set the fixed parameter order (tuple of names). Call once at setup."""
        import jax
        self._order = tuple(order)
        row_order = self._order
        base = self._fn  # already jitted above

        def row_to_scalar(row):
            # Dict is built here during tracing; not per runtime call.
            params = {name: row[i] for i, name in enumerate(row_order)}
            return base(params)  # scalar logL

        self._row = jax.jit(row_to_scalar)
        self._mat = jax.jit(jax.vmap(self._row))

    def log_likelihood_row(self, row):
        return self._row(row)

    def log_likelihood_matrix(self, X):
        return self._mat(X)

# ---------------------------- Prior parsing ---------------------------- #

class PriorParsingError(ValueError):
    pass


def _parse_single_prior(name: str, spec: PriorSpec) -> ParsedPrior:
    # Shorthand tuple forms
    if isinstance(spec, tuple):
        tag = spec[0].lower()
        if tag == "uniform":
            _, a, b = spec
            return ParsedPrior("uniform", (float(a), float(b)), {"min": float(a), "max": float(b)})
        if tag == "loguniform":
            _, a, b = spec
            return ParsedPrior("loguniform", (float(a), float(b)), {"a": float(a), "b": float(b)})
        if tag == "normal":
            # Allow ('normal', mean, sigma) or ('normal', mean, sigma, _)
            if len(spec) == 3:
                _, mu, sigma = spec
            else:
                _, mu, sigma, _unused = spec
            return ParsedPrior("normal", None, {"mean": float(mu), "sigma": float(sigma)})
        if tag == "fixed":
            _, val = spec
            return ParsedPrior("fixed", None, {"value": float(val)})
        raise PriorParsingError(f"Unsupported prior tuple for {name}: {spec}")

    # Dict-like
    if isinstance(spec, Mapping):
        if "dist" not in spec:
            # Callable prior with explicit bounds
            if callable(spec.get("logpdf")):
                bounds = spec.get("bounds")
                if bounds is None:
                    raise PriorParsingError(
                        f"Callable prior for {name} requires 'bounds'=(min,max)."
                    )
                return ParsedPrior("callable", (float(bounds[0]), float(bounds[1])), {}, logpdf=spec["logpdf"])            
            raise PriorParsingError(f"Prior for {name} missing 'dist' key: {spec}")
        dist = str(spec["dist"]).lower()
        if dist == "uniform":
            a, b = spec.get("min"), spec.get("max")
            if a is None or b is None:
                raise PriorParsingError(f"Uniform prior for {name} requires 'min' and 'max'")
            return ParsedPrior("uniform", (float(a), float(b)), {"min": float(a), "max": float(b)})
        if dist == "loguniform":
            a, b = spec.get("a"), spec.get("b")
            if a is None or b is None:
                raise PriorParsingError(f"Log-uniform prior for {name} requires 'a' and 'b'")
            return ParsedPrior("loguniform", (float(a), float(b)), {"a": float(a), "b": float(b)})
        if dist == "normal":
            mu, sigma = spec.get("mean"), spec.get("sigma")
            if mu is None or sigma is None:
                raise PriorParsingError(f"Normal prior for {name} requires 'mean' and 'sigma'")
            # Normal is unbounded; user can add 'bounds' if desired
            bounds = spec.get("bounds")
            bnd = (float(bounds[0]), float(bounds[1])) if bounds is not None else None
            return ParsedPrior("normal", bnd, {"mean": float(mu), "sigma": float(sigma)})
        if dist == "fixed":
            val = spec.get("value")
            if val is None:
                raise PriorParsingError(f"Fixed prior for {name} requires 'value'")
            return ParsedPrior("fixed", None, {"value": float(val)})
        raise PriorParsingError(f"Unsupported prior dist '{dist}' for {name}")

    raise PriorParsingError(f"Unsupported prior spec for {name}: {spec}")


def _split_priors(prior: Mapping[ParamName, PriorSpec]):
    """Split priors into sampled and fixed parameters and derive bounds.

    Returns
    -------
    sampled_names : list[str]
    fixed : dict[str, float]
    bounds : dict[str, tuple[float, float]]
    logprior_fns : dict[str, Callable[[float], float]]
    """
    sampled_names: List[str] = []
    fixed: Dict[str, float] = {}
    bounds: Dict[str, Tuple[float, float]] = {}
    logprior_fns: Dict[str, Callable[[float], float]] = {}

    for name, spec in prior.items():
        p = _parse_single_prior(name, spec)
        if p.dist == "fixed":
            fixed[name] = p.params["value"]
            continue
        sampled_names.append(name)
        # Bounds are required by nessai for sampling. For normal, require explicit bounds.
        if p.bounds is None:
            if p.dist == "normal":
                # Set a standard 5-sigma bound if none provided
                print(f"Warning: setting 5-sigma bounds for normal prior '{name}'")
                p.bounds = (-5 * p.params["sigma"] + p.params["mean"], 5 * p.params["sigma"] + p.params["mean"])
            else:
                raise PriorParsingError(f"Bounds are required for prior '{name}' with dist {p.dist}.")
        bounds[name] = (float(p.bounds[0]), float(p.bounds[1]))

        # Build logpdf callable per parameter
        if p.dist == "uniform":
            a, b = p.params["min"], p.params["max"]
            width = b - a
            const = -math.log(width)
            def make_uniform(a=a, b=b, const=const):
                def _logpdf(x: float) -> float:
                    return const if (a <= x <= b) else -math.inf
                return _logpdf
            logprior_fns[name] = make_uniform()
        elif p.dist == "loguniform":
            a, b = p.params["a"], p.params["b"]
            if a <= 0 or b <= 0:
                raise PriorParsingError(f"Log-uniform bounds must be positive for {name}")
            c = math.log(a)
            norm = -math.log(math.log(b) - math.log(a))
            def make_loguni(a=a, b=b, c=c, norm=norm):
                def _logpdf(x: float) -> float:
                    if x <= 0 or x < a or x > b:
                        return -math.inf
                    return -math.log(x) + norm
                return _logpdf
            logprior_fns[name] = make_loguni()
        elif p.dist == "normal":
            mu, sigma = p.params["mean"], p.params["sigma"]
            const = -0.5 * math.log(2 * math.pi * sigma * sigma)
            def make_norm(mu=mu, sigma=sigma, const=const, b=p.bounds):
                lo, hi = b if b is not None else (-math.inf, math.inf)
                def _logpdf(x: float) -> float:
                    if not (lo <= x <= hi):
                        return -math.inf
                    z = (x - mu) / sigma
                    return const - 0.5 * z * z
                return _logpdf
            logprior_fns[name] = make_norm()
        elif p.dist == "callable":
            if p.logpdf is None:
                raise PriorParsingError(f"Callable prior for {name} missing logpdf")
            b = p.bounds
            lo, hi = b if b is not None else (-math.inf, math.inf)
            def make_callable(lp=p.logpdf, lo=lo, hi=hi):
                def _logpdf(x: float) -> float:
                    if not (lo <= x <= hi):
                        return -math.inf
                    return float(lp(x))
                return _logpdf
            logprior_fns[name] = make_callable()
        else:
            raise PriorParsingError(f"Unsupported prior dist '{p.dist}' while building logpdf for {name}")

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
    discovery_adapter : _DiscoveryAdapter
        Adapter to call the Discovery model with a parameter dict.
    """
    def __init__(self,
                 names: List[str],
                 bounds: Dict[str, Tuple[float, float]],
                 logprior_fns: Dict[str, Callable[[float], float]],
                 fixed_params: Dict[str, float],
                 discovery_adapter: _DiscoveryAdapter):
        super().__init__()
        self.names = list(names)                      
        self._names_tuple = tuple(self.names)          # optional: faster internal iteration

        self.bounds = dict(bounds)
        self._logprior_fns = dict(logprior_fns)
        self._fixed = dict(fixed_params)
        self._adapter = discovery_adapter

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

    def __log_likelihood(self, x: np.ndarray) -> np.ndarray:
        xb = _as_batch_struct(x)                 # shape (N,)
        N = xb.shape[0]

        # Build a params dict of DEVICE arrays (one leaf per parameter).
        # Important: avoid Python float() casts, they pull values to host.
        params = {n: jnp.asarray(xb[n]) for n in self.names}

        # Broadcast fixed params to shape (N,)
        if self._fixed:
            params.update({k: jnp.full((N,), float(v)) for k, v in self._fixed.items()})

        # One batched call; returns shape (N,)
        return self._adapter.log_likelihood_batch(params)

    def old_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        xb = _as_batch_struct(x)                   # structured np array with shape (N,)
        N = xb.shape[0]

        # 1) Build ONE dense host array (N, D_s) from the structured array.
        # This is vectorised C code; usually zero-copy view when dtypes are uniform.
        X = rfn.structured_to_unstructured(xb[list(self.names)], dtype=np.float64)
        # (N, D_s). Ensure contiguous, then ONE device transfer:
        Xj = jnp.asarray(np.ascontiguousarray(X), dtype=jnp.float64)

        # 2) Append fixed params as broadcast columns (on device, no Python loop per-row)
        if self._fixed:
            fvec = jnp.array([float(self._fixed[k]) for k in self._fixed], dtype=jnp.float64)   # (D_f,)
            F = jnp.broadcast_to(fvec, (N, fvec.size))                                          # (N, D_f)
            Xj = jnp.concatenate([Xj, F], axis=1)                                               # (N, D)

        # 3) Cheap column views to rebuild dict-of-arrays (no device_puts here)
        params = {n: Xj[:, i] for i, n in enumerate(self._all_names)}

        # 4) Single batched call (vmap+jit inside adapter)
        return self._adapter.log_likelihood_batch(params)

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
        self.adapter = _DiscoveryAdapter(discovery_model, jit=jit)
        snames, fixed, bounds, lpfns = _split_priors(priors)
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
                    seed: Optional[int] = None,
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
            seed=seed,
            resume=resume,
            **kwargs,
        )
        self.results = self.sampler.run()

        return self.results
    

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