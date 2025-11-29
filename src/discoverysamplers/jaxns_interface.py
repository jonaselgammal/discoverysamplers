
"""
Discovery ↔︎ JAX-NS Interface

This module provides a bridge between Discovery-style models and JAX-NS nested sampling.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import math

import numpy as np
import numpy.lib.recfunctions as rfn

from .priors import ParsedPrior, PriorParsingError, _parse_single_prior, _split_priors, ParamName, PriorSpec
from .likelihood import LikelihoodWrapper

# --------------------------- JAX-NS Bridge ----------------------------- #

try:
    # Only needed when actually running the sampler
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erfinv
    # jaxns imports are version dependent; we try common entry points.
    import jaxns as jns  # type: ignore
    from jaxns import NestedSampler  # type: ignore
    from jaxns import Model, Prior  # type: ignore
    import tensorflow_probability.substrates.jax as tfp  # type: ignore
    tfpd = tfp.distributions  # type: ignore
except Exception:  # pragma: no cover - allow import without jaxns installed
    jns = None      # type: ignore
    NestedSampler = None  # type: ignore
    Model = None  # type: ignore
    Prior = None  # type: ignore
    tfpd = None  # type: ignore

# -------------------------- JAX-NS specific utilities ------------------- #


def _ndtri(u: "jnp.ndarray") -> "jnp.ndarray":
    """Inverse Normal CDF using erfinv: Φ^{-1}(u) = sqrt(2) * erfinv(2u-1)."""
    return jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)


def _make_prior_transform(sampled_names, parsed_specs_dict):
    """
    Build a vectorised prior transform f: U in [0,1]^D -> theta in R^D,
    matching the order in `sampled_names`.

    `parsed_specs_dict` is a dict[name] -> ParsedPrior from _parse_single_prior.
    This function adapts the common ParsedPrior format to JAX-NS requirements.
    """
    # Precompute constants for speed; everything JAX-friendly
    kinds = []
    params_list = []
    for n in sampled_names:
        p = parsed_specs_dict[n]
        kinds.append(p.dist_type)

        # Convert ParsedPrior to parameter dict for JAX-NS
        if p.dist_type == "uniform":
            params_list.append({"min": p.bounds[0], "max": p.bounds[1]})
        elif p.dist_type == "loguniform":
            params_list.append({"a": p.bounds[0], "b": p.bounds[1]})
        elif p.dist_type == "normal":
            params_list.append({"mean": p.mean, "sigma": p.sigma})
        else:
            raise ValueError(
                f"JAX-NS bridge: prior '{n}' with dist '{p.dist_type}' "
                "is not supported for transform-based sampling. "
                "Use a standard distribution (uniform/loguniform/normal)."
            )

    kinds_tuple = tuple(kinds)  # for Python-side dispatch

    def transform(uvec: "jnp.ndarray") -> "jnp.ndarray":
        # uvec shape (..., D); returns same leading shape with D parameters
        def one(u):
            outs = []
            for i, kind in enumerate(kinds_tuple):
                ui = u[i]
                par = params_list[i]
                if kind == "uniform":
                    a = par["min"]; b = par["max"]
                    outs.append(a + ui * (b - a))
                elif kind == "loguniform":
                    a = par["a"]; b = par["b"]
                    # exp( log(a) + u*(log(b)-log(a)) )
                    outs.append(jnp.exp(jnp.log(a) + ui * (jnp.log(b) - jnp.log(a))))
                elif kind == "normal":
                    mu = par["mean"]; sigma = par["sigma"]
                    outs.append(mu + sigma * _ndtri(ui))
                else:
                    # Should never reach here due to check above
                    raise ValueError(
                        f"JAX-NS bridge: prior '{sampled_names[i]}' with dist '{kind}' "
                        "is not supported for transform-based sampling."
                    )
            return jnp.stack(outs, axis=0)

        uvec = jnp.asarray(uvec, dtype=jnp.float64)
        if uvec.ndim == 1:
            return one(uvec)
        return jax.vmap(one)(uvec)

    return transform


class DiscoveryJAXNSBridge:
    """
    Bridge between a Discovery-style model and JAX-NS NestedSampler.

    Mirrors DiscoveryNessaiBridge API where possible.

    Parameters
    ----------
    discovery_model : callable | object
        Callable or object with `.logL(params_dict) -> float`.
    priors : Mapping[str, PriorSpec]
        Same schema you use for the nessai bridge.
    latex_labels : Optional[Mapping[str, str]]
        Optional labels used for plotting/exports.
    jit : bool
        JIT the discovery model for fast likelihood calls.
    """

    def __init__(self,
                 discovery_model,
                 priors,
                 latex_labels=None,
                 jit: bool = True):
        if jns is None or NestedSampler is None:
            raise RuntimeError("jaxns is not installed. Please `pip install jaxns`.")

        # Parse priors once; keep the order from the input dict
        sampled_names, fixed, bounds, _logprior_fns = _split_priors(priors)
        if not sampled_names:
            raise ValueError("No sampled parameters defined (all fixed?)")

        # Reuse your adapter and prior splitter - pass fixed params to adapter
        self.adapter = LikelihoodWrapper(discovery_model, jit=jit, fixed_params=fixed, allow_array_api=True)

        self.sampled_names = list(sampled_names)
        self.fixed_params = dict(fixed)
        self.bounds = dict(bounds)              # not strictly used by JAX-NS, but kept for parity
        self.discovery_paramnames = list(priors.keys())
        self.fixed_names = list(self.fixed_params.keys())

        self.latex_labels = dict(latex_labels) if latex_labels else {n: n for n in self.discovery_paramnames}
        self.latex_list = [self.latex_labels.get(n, n) for n in self.discovery_paramnames]
        self.sampled_names_latex = [self.latex_labels.get(n, n) for n in self.sampled_names]
        self.fixed_names_latex = [self.latex_labels.get(n, n) for n in self.fixed_names]

        # Build a map name -> ParsedPrior to feed the transform
        # Use the common parser for all parameters
        parsed_specs = {n: _parse_single_prior(n, priors[n]) for n in priors}
        self._parsed_sampled = {n: parsed_specs[n] for n in self.sampled_names}

        # Prior transform over unit-cube -> sampled parameter vector (in the sampled order)
        self._prior_transform_vec = _make_prior_transform(self.sampled_names, self._parsed_sampled)

        # Configure the discovery adapter’s array API (so the hot path is vectorised)
        all_order = tuple(self.sampled_names) + tuple(self.fixed_names)
        if hasattr(self.adapter, "configure_array_api"):
            self.adapter.configure_array_api(all_order)

        # Likelihood wrappers ------------------------------------------------
        # Vector form: theta_sampled (D_s,)  -> scalar logL
        # Expand with fixed params then call compiled row function.
        def _loglik_row(theta_s: "jnp.ndarray") -> "jnp.float64":
            vals = [theta_s[i] for i in range(len(self.sampled_names))]
            if self.fixed_params:
                vals.extend([self.fixed_params[k] for k in self.fixed_params])
            row = jnp.asarray(jnp.array(vals, dtype=jnp.float64))
            return self.adapter.log_likelihood_row(row)

        # Batch form: map over leading axis with vmap
        self._loglik_row = jax.jit(_loglik_row)
        self._loglik_batch = jax.jit(jax.vmap(self._loglik_row))

        # JAX-NS API expects:
        #   - loglikelihood(theta) that accepts theta either shape (D,) or (N,D)
        #   - prior_transform(u) from unit-cube to theta, with the same shape contract
        def _loglik(theta):
            theta = jnp.asarray(theta, dtype=jnp.float64)
            if theta.ndim == 1:
                return self._loglik_row(theta)
            return self._loglik_batch(theta)

        def _prior_transform(u):
            return self._prior_transform_vec(u)

        self._jaxns_loglik = _loglik
        self._jaxns_prior_transform = _prior_transform

        # Sampler placeholder
        self.ns = None
        self.results = None

    # ---------------------------- Running ------------------------------- #
    def run_sampler(self,
                    *,
                    nlive: int = 800,
                    max_samples: Optional[int] = None,
                    termination_frac: float = 0.01,
                    rng_seed: Optional[int] = None,
                    sampler_kwargs: Optional[dict] = None):
        """
        Run JAX-NS NestedSampler.

        Parameters
        ----------
        nlive : int
            Number of live points.
        max_samples : int | None
            Optional hard cap on the number of samples (pass through when supported).
        termination_frac : float
            Evidence tolerance fraction at which to terminate (version dependent).
        rng_seed : int | None
            Seed for JAX PRNG.
        sampler_kwargs : dict | None
            Extra kwargs forwarded to NestedSampler (e.g., sampler implementation).

        Returns
        -------
        results : object
            JAX-NS results object (version-dependent structure).
        """
        if rng_seed is None:
            rng_seed = 0
        key = jax.random.PRNGKey(rng_seed)

        sampler_kwargs = dict(sampler_kwargs or {})
        
        # Create JAX-NS Model (new API)
        # prior_model must be a generator function that yields priors
        def prior_model():
            priors_list = []
            for name in self.sampled_names:
                parsed = self._parsed_sampled[name]
                if parsed.dist_type == "uniform":
                    prior = Prior(tfpd.Uniform(
                        low=parsed.bounds[0], 
                        high=parsed.bounds[1]
                    ), name=name)
                elif parsed.dist_type == "loguniform":
                    # LogUniform = exp(Uniform(log(a), log(b)))
                    prior = Prior(tfpd.TransformedDistribution(
                        distribution=tfpd.Uniform(
                            low=jnp.log(parsed.bounds[0]),
                            high=jnp.log(parsed.bounds[1])
                        ),
                        bijector=tfp.bijectors.Exp()
                    ), name=name)
                elif parsed.dist_type == "normal":
                    prior = Prior(tfpd.Normal(
                        loc=parsed.mean,
                        scale=parsed.sigma
                    ), name=name)
                else:
                    raise ValueError(f"Unsupported prior type '{parsed.dist_type}' for parameter '{name}'")
                priors_list.append((yield prior))
            return tuple(priors_list)
        
        # Define likelihood function that takes unpacked sampled parameter values
        def log_likelihood(*sampled_params):
            # sampled_params are unpacked values in the same order as self.sampled_names
            # Build dict with only sampled params - adapter will add fixed params
            sampled_dict = {}
            for i, name in enumerate(self.sampled_names):
                sampled_dict[name] = sampled_params[i]
            # Adapter will automatically merge with fixed_params
            return self.adapter.log_likelihood(sampled_dict)
        
        # Create the model
        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        
        # Instantiate the sampler with the model
        self.ns = NestedSampler(
            model=model,
            max_samples=max_samples,
            num_live_points=nlive,
            **sampler_kwargs,
        )

        # Run the sampler
        termination_kwargs = {}
        if termination_frac is not None:
            # Try to pass termination_frac if supported
            termination_kwargs["termination_frac"] = float(termination_frac)
        
        try:
            raw_results = self.ns(key, **termination_kwargs)
        except TypeError:
            # If termination_frac not supported, run without it
            raw_results = self.ns(key)
        
        # Process results - newer JAXNS returns (termination_reason, state) tuple
        if isinstance(raw_results, tuple) and len(raw_results) == 2:
            termination_reason, state = raw_results
            # Compute log evidence manually from nested samples
            # Simple trapezoid rule estimation
            # IMPORTANT: state.num_samples can be larger than the actual array size,
            # so we use the minimum of num_samples and the actual array length
            log_L_full = state.sample_collection.log_L
            n = min(int(state.num_samples), log_L_full.shape[0])
            
            log_L = log_L_full[:n]
            # Sort by log-likelihood
            sorted_idx = jnp.argsort(log_L)
            log_L_sorted = log_L[sorted_idx]
            
            # Compute log prior mass (uniform spacing in nested sampling)
            log_X = jnp.log(jnp.linspace(1.0, 1.0/n, n))
            
            # Trapezoid rule for log evidence
            log_widths = jnp.diff(jnp.concatenate([jnp.array([0.0]), log_X]))
            
            # Compute log(exp(log_L) * width) = log_L + log(width)
            log_weights = log_L_sorted + jnp.log(-log_widths)
            # LogSumExp to get total evidence
            from jax.scipy.special import logsumexp
            log_Z = logsumexp(log_weights)
            # Rough uncertainty estimate (1/sqrt(n))
            log_Z_uncert = 1.0 / jnp.sqrt(float(n))
            
            # Create a results dict for compatibility
            self.results = {
                'termination_reason': termination_reason,
                'state': state,
                'logZ': float(log_Z),
                'logZerr': float(log_Z_uncert),
                'log_Z_mean': float(log_Z),
                'log_Z_uncert': float(log_Z_uncert),
                'samples': state.sample_collection,
            }
        else:
            # Older API or different structure
            self.results = raw_results
        
        return self.results

    # --------------- Results extraction & convenience ------------------ #
    def _posterior_from_results(self, results=None):
        """
        Try to obtain an equal-weight set of posterior draws (nsamples, D_s).
        Returns (chain_sampled, weight_array_or_None).
        """
        res = results if results is not None else self.results
        if res is None:
            raise RuntimeError("No results available. Run `run_sampler()` first.")

        def _as_ndarray(x):
            try:
                return np.asarray(x, dtype=float)
            except Exception:
                return None

        chain = None
        
        # Handle new API: results dict with 'state' containing sample_collection
        if isinstance(res, Mapping) and 'state' in res:
            state = res['state']
            if hasattr(state, 'sample_collection'):
                sc = state.sample_collection
                
                # Get actual number of samples (min of num_samples and array size)
                n = sc.log_L.shape[0]
                if hasattr(state, 'num_samples'):
                    n = min(int(state.num_samples), n)
                
                # U_samples contains the unit hypercube samples
                # Shape is (n_samples, n_params) - already in the correct format
                if hasattr(sc, 'U_samples'):
                    U_samples = sc.U_samples
                    U_samples_np = np.asarray(U_samples, dtype=float)[:n]
                    
                    # Transform from unit hypercube to physical space
                    # For each parameter, apply the inverse CDF of the prior
                    chain = np.zeros_like(U_samples_np)
                    for i, name in enumerate(self.sampled_names):
                        parsed = self._parsed_sampled[name]
                        u = U_samples_np[:, i]
                        if parsed.dist_type == 'uniform':
                            low, high = parsed.bounds
                            chain[:, i] = low + u * (high - low)
                        elif parsed.dist_type == 'loguniform':
                            low, high = parsed.bounds
                            log_low, log_high = np.log(low), np.log(high)
                            chain[:, i] = np.exp(log_low + u * (log_high - log_low))
                        elif parsed.dist_type == 'normal':
                            from scipy import stats
                            chain[:, i] = stats.norm.ppf(u, loc=parsed.mean, scale=parsed.sigma)
        
        # Common possibilities across versions:
        # - res.samples or res['samples']  -> dict with 'theta' or ndarray
        # - res.posterior_samples          -> ndarray
        # - res['posterior']['samples']    -> ndarray
        # If equal-weight posterior utility exists, prefer that.
        # We duck-type check a few.
        # 1) Equal-weight helper (if present)
        if chain is None:
            for attr in ("equal_weight_posterior", "get_equal_weight_posterior", "posterior_equal_weights"):
                fn = getattr(self.ns, attr, None)
                if callable(fn):
                    key = jax.random.PRNGKey(0)
                    try:
                        arr = fn(res, key=key) if "key" in fn.__code__.co_varnames else fn(res)
                        if isinstance(arr, (np.ndarray, jnp.ndarray)):
                            chain = np.asarray(arr, dtype=float)
                            break
                    except Exception:
                        pass

        # 2) Look for common fields
        if chain is None:
            candidates = []
            for k in ("posterior_samples", "samples", "theta", "posterior"):
                val = getattr(res, k, None)
                if val is not None:
                    candidates.append(val)
                if isinstance(res, Mapping) and k in res:
                    candidates.append(res[k])
            for c in candidates:
                arr = _as_ndarray(c)
                if arr is not None and arr.ndim == 2 and arr.shape[1] == len(self.sampled_names):
                    chain = arr
                    break

        if chain is None:
            raise RuntimeError(
                "Could not locate posterior samples in JAX-NS results. "
                "If your version exposes a different field, pass it to "
                "`return_sampled_samples(results=...)`."
            )
        weights = None
        return chain, weights

    def _stack_with_fixed(self, sampled_chain: np.ndarray) -> np.ndarray:
        """Combine sampled columns with fixed params in original priors order."""
        nsamp = sampled_chain.shape[0]
        chain_all = np.zeros((nsamp, len(self.discovery_paramnames)), dtype=float)
        # fill sampled
        for j, name in enumerate(self.sampled_names):
            idx = self.discovery_paramnames.index(name)
            chain_all[:, idx] = sampled_chain[:, j]
        # fill fixed
        for name in self.fixed_names:
            idx = self.discovery_paramnames.index(name)
            chain_all[:, idx] = float(self.fixed_params[name])
        return chain_all

    # Public API (mirrors nessai bridge) ---------------------------------
    def return_sampled_samples(self, *, results=None) -> Dict[str, Any]:
        chain, _ = self._posterior_from_results(results)
        return {"names": self.sampled_names,
                "labels": self.sampled_names_latex,
                "chain": np.asarray(chain, dtype=float)}

    def return_all_samples(self, *, results=None) -> Dict[str, Any]:
        chain_s, _ = self._posterior_from_results(results)
        chain_all = self._stack_with_fixed(chain_s)
        return {"names": self.discovery_paramnames,
                "labels": self.latex_list,
                "chain": chain_all}

    def return_logZ(self, *, results=None) -> Dict[str, float]:
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
        
        logZ = None
        logZ_err = None
        
        # Check common keys in results dict
        for key in ('logZ', 'log_Z_mean', 'log_evidence', 'log_Z'):
            if isinstance(res, Mapping) and key in res:
                logZ = float(res[key])
                break
        
        for key in ('logZerr', 'logZ_err', 'log_Z_uncert', 'log_evidence_error', 'log_Z_err'):
            if isinstance(res, Mapping) and key in res:
                logZ_err = float(res[key])
                break
        
        # Try object attributes if dict keys failed
        if logZ is None:
            for attr in ('log_Z_mean', 'logZ', 'log_evidence'):
                if hasattr(res, attr):
                    logZ = float(getattr(res, attr))
                    break
        
        if logZ_err is None:
            for attr in ('log_Z_uncert', 'logZerr', 'log_evidence_error'):
                if hasattr(res, attr):
                    logZ_err = float(getattr(res, attr))
                    break
        
        if logZ is None:
            raise RuntimeError("Could not find log evidence in results. Check that sampling completed successfully.")
        
        return {'logZ': logZ, 'logZ_err': logZ_err}

    # ------------------------------ Plots ------------------------------ #
    def plot_trace(self, *, burn: int = 0, plot_fixed: bool = False, results=None, **kwargs):
        """
        Plot trace of samples vs sample index.
        
        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        plot_fixed : bool, optional
            If True, includes fixed parameters in the plot, by default False.
        results : optional
            Results from run_sampler(). If None, uses stored results.
        **kwargs
            Additional keyword arguments passed to plots.plot_trace().
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the trace plots.
        """
        from .plots import plot_trace

        data = self.return_all_samples(results=results) if plot_fixed else self.return_sampled_samples(results=results)
        return plot_trace(
            data, 
            burn=burn,
            fixed_params=self.fixed_params,
            fixed_names=self.fixed_names,
            **kwargs
        )

    def plot_corner(self, *, burn: int = 0, results=None, **kwargs):
        """
        Corner plot of sampled parameters.
        
        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        results : optional
            Results from run_sampler(). If None, uses stored results.
        **kwargs
            Additional keyword arguments passed to corner.corner().
            
        Returns
        -------
        matplotlib.figure.Figure
            Corner plot figure.
        """
        from .plots import plot_corner

        data = self.return_sampled_samples(results=results)
        return plot_corner(data, burn=burn, **kwargs)
