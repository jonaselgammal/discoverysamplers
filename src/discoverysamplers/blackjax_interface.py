"""
Discovery ↔ BlackJAX Nested Sampling Interface

This module provides a bridge between Discovery-style models and BlackJAX's
nested sampling implementation (from handley-lab/blackjax branch).

Key features:
- Adapts Discovery dict-based likelihoods to BlackJAX array-based interface
- Handles prior transformations from unit cube to parameter space
- Supports JAX JIT compilation for performance
- Compatible with BlackJAX's nested sampling API (nss algorithm)

Installation
------------
pip install git+https://github.com/handley-lab/blackjax.git@nested_sampling

API Reference
-------------
The BlackJAX nested sampling API from handley-lab/blackjax:
- blackjax.nss(logprior_fn, loglikelihood_fn, num_inner_steps, num_delete=1, ...)
- algo.init(particles) -> NSState
- algo.step(rng_key, state) -> (NSState, NSInfo)
- blackjax.ns.utils.finalise(live_state, dead_list) -> NSInfo with combined samples
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import warnings
import os

import numpy as np
import jax
import jax.numpy as jnp

from .priors import ParsedPrior, PriorParsingError, _parse_single_prior, _split_priors, ParamName, PriorSpec
from .likelihood import LikelihoodWrapper

try:
    import blackjax
    import blackjax.ns.utils as ns_utils
    BLACKJAX_AVAILABLE = True
except ImportError:
    blackjax = None
    ns_utils = None
    BLACKJAX_AVAILABLE = False
    warnings.warn(
        "BlackJAX not available. Please install from: "
        "pip install git+https://github.com/handley-lab/blackjax.git@nested_sampling",
        RuntimeWarning
    )


def _make_prior_transform_blackjax(sampled_names: List[str],
                                    parsed_priors: Dict[str, ParsedPrior]) -> Callable:
    """
    Build a prior transform that maps from unit cube [0,1]^n to parameter space.

    Parameters
    ----------
    sampled_names : list of str
        Parameter names in order
    parsed_priors : dict
        Mapping of parameter name to ParsedPrior object

    Returns
    -------
    callable
        Function that transforms unit cube samples to physical parameters
    """
    transforms = []

    for name in sampled_names:
        prior = parsed_priors[name]

        if prior.dist_type == 'uniform':
            a, b = prior.bounds
            transforms.append(lambda u, a=a, b=b: a + u * (b - a))

        elif prior.dist_type == 'loguniform':
            a, b = prior.bounds
            # Pre-compute as Python floats to avoid Metal issues
            log_a, log_b = float(np.log(a)), float(np.log(b))
            transforms.append(lambda u, log_a=log_a, log_b=log_b: jnp.exp(log_a + u * (log_b - log_a)))

        elif prior.dist_type == 'normal':
            mean, sigma = prior.mean, prior.sigma
            # Use inverse normal CDF (erfinv)
            from jax.scipy.special import erfinv
            sqrt2 = float(np.sqrt(2.0))
            if prior.bounds is None:
                # Unbounded normal
                transforms.append(lambda u, mean=mean, sigma=sigma, sqrt2=sqrt2:
                                mean + sigma * sqrt2 * erfinv(2.0 * u - 1.0))
            else:
                # Truncated normal - approximate with uniform for now
                a, b = prior.bounds
                warnings.warn(f"Truncated normal prior for {name} approximated as uniform")
                transforms.append(lambda u, a=a, b=b: a + u * (b - a))
        else:
            raise ValueError(f"Unsupported prior type for nested sampling: {prior.dist_type}")

    @jax.jit
    def prior_transform(u: jnp.ndarray) -> jnp.ndarray:
        """Transform unit cube sample to physical parameters."""
        return jnp.array([transforms[i](u[i]) for i in range(len(u))])

    return prior_transform


def _make_log_prior_blackjax(sampled_names: List[str],
                              parsed_priors: Dict[str, ParsedPrior]) -> Callable:
    """
    Build a log-prior function for physical parameters.

    Parameters
    ----------
    sampled_names : list of str
        Parameter names in order
    parsed_priors : dict
        Mapping of parameter name to ParsedPrior object

    Returns
    -------
    callable
        Function that computes log-prior density
    """
    logprior_fns = []

    for name in sampled_names:
        prior = parsed_priors[name]

        if prior.dist_type == 'uniform':
            a, b = prior.bounds
            # Pre-compute log width as Python float to avoid Metal issues
            log_width = float(np.log(b - a))
            logprior_fns.append(lambda x, a=a, b=b, lw=log_width:
                              jnp.where((x >= a) & (x <= b), -lw, -jnp.inf))

        elif prior.dist_type == 'loguniform':
            a, b = prior.bounds
            # Pre-compute as Python floats
            log_a, log_b = float(np.log(a)), float(np.log(b))
            norm = float(np.log(log_b - log_a))
            logprior_fns.append(lambda x, a=a, b=b, log_a=log_a, norm=norm:
                              jnp.where((x >= a) & (x <= b), -jnp.log(x) - norm, -jnp.inf))

        elif prior.dist_type == 'normal':
            mean, sigma = prior.mean, prior.sigma
            # Pre-compute as Python float
            log_norm = float(0.5 * np.log(2 * np.pi * sigma**2))
            if prior.bounds is None:
                logprior_fns.append(lambda x, mean=mean, sigma=sigma, ln=log_norm:
                                  -0.5 * ((x - mean) / sigma)**2 - ln)
            else:
                a, b = prior.bounds
                logprior_fns.append(lambda x, mean=mean, sigma=sigma, a=a, b=b, ln=log_norm:
                                  jnp.where((x >= a) & (x <= b),
                                          -0.5 * ((x - mean) / sigma)**2 - ln,
                                          -jnp.inf))

    @jax.jit
    def log_prior(theta: jnp.ndarray) -> float:
        """Compute log-prior for parameter vector."""
        logp = 0.0
        for i, fn in enumerate(logprior_fns):
            logp += fn(theta[i])
        return logp

    return log_prior


class DiscoveryBlackJAXBridge:
    """
    Bridge between Discovery models and BlackJAX nested sampling.

    This class adapts Discovery's dict-based likelihood interface to BlackJAX's
    array-based nested sampling implementation from the handley-lab fork.

    Parameters
    ----------
    discovery_model : callable
        Discovery model likelihood function (e.g., likelihood.logL)
    priors : dict
        Prior specifications for parameters
    latex_labels : dict, optional
        LaTeX labels for parameters (for plotting)
    jit : bool, default=True
        Whether to JIT compile the likelihood

    Examples
    --------
    >>> import discovery as ds
    >>> from discoverysamplers.blackjax_interface import DiscoveryBlackJAXBridge
    >>>
    >>> # Create Discovery likelihood
    >>> psr = ds.Pulsar.read_feather('pulsar.feather')
    >>> likelihood = ds.PulsarLikelihood([...])
    >>>
    >>> # Define priors
    >>> priors = {
    ...     'mass': ('uniform', 1.0, 3.0),
    ...     'distance': ('loguniform', 0.1, 100.0),
    ... }
    >>>
    >>> # Create bridge
    >>> bridge = DiscoveryBlackJAXBridge(likelihood.logL, priors)
    >>>
    >>> # Run sampler
    >>> results = bridge.run_sampler(n_live=500, max_iterations=10000)

    Notes
    -----
    This uses the nested slice sampling (NSS) algorithm from handley-lab/blackjax.
    The algorithm uses Hit-and-Run Slice Sampling (HRSS) as its inner kernel,
    with adaptive tuning of the covariance matrix used for slice directions.
    """

    def __init__(self,
                 discovery_model: Any,
                 priors: Mapping[str, PriorSpec],
                 latex_labels: Optional[Mapping[str, str]] = None,
                 jit: bool = True):

        if not BLACKJAX_AVAILABLE:
            raise RuntimeError(
                "BlackJAX is not installed. Please install from: "
                "pip install git+https://github.com/handley-lab/blackjax.git@nested_sampling"
            )

        self.latex_labels = latex_labels or {}

        # Wrap the Discovery model
        self.adapter = LikelihoodWrapper(discovery_model, jit=jit, fixed_params=None, allow_array_api=False)

        # Parse priors
        sampled_names, fixed_params, bounds, logprior_fns = _split_priors(priors)

        if not sampled_names:
            raise ValueError("No sampled parameters defined (all fixed?)")

        self.sampled_names = sampled_names
        self.fixed_params = fixed_params
        self.bounds = bounds
        self.ndim = len(sampled_names)

        # Store original priors order for output formatting
        self.discovery_paramnames = list(priors.keys())
        self.fixed_names = list(fixed_params.keys())

        # Build latex label lists
        self.latex_list = [self.latex_labels.get(n, n) for n in self.discovery_paramnames]
        self.sampled_names_latex = [self.latex_labels.get(n, n) for n in self.sampled_names]
        self.fixed_names_latex = [self.latex_labels.get(n, n) for n in self.fixed_names]

        # Parse priors into ParsedPrior objects
        self.parsed_priors = {name: _parse_single_prior(name, priors[name])
                              for name in sampled_names}

        # Build prior transform and log-prior
        self.prior_transform = _make_prior_transform_blackjax(sampled_names, self.parsed_priors)
        self.log_prior_fn = _make_log_prior_blackjax(sampled_names, self.parsed_priors)

        # Store results
        self.results = None
        self.sampler = None
        self._dead = None  # Store dead points for post-processing

    def _theta_to_dict(self, theta: jnp.ndarray) -> Dict[str, float]:
        """Convert parameter array to dict for Discovery model."""
        param_dict = {name: theta[i] for i, name in enumerate(self.sampled_names)}
        param_dict.update(self.fixed_params)
        return param_dict

    @property
    def loglikelihood_fn(self) -> Callable:
        """Log-likelihood function for BlackJAX (takes array, returns scalar)."""
        def loglike(theta: jnp.ndarray) -> float:
            param_dict = self._theta_to_dict(theta)
            return self.adapter.log_likelihood(param_dict)
        return loglike

    def run_sampler(self,
                    n_live: int = 500,
                    max_iterations: Optional[int] = None,
                    num_delete: int = 10,
                    num_inner_steps: Optional[int] = None,
                    termination_threshold: float = -3.0,
                    rng_key: Optional[Any] = None,
                    seed: int = 0,
                    progress: bool = True) -> Dict[str, Any]:
        """
        Run BlackJAX nested sampling.

        Parameters
        ----------
        n_live : int, default=500
            Number of live points
        max_iterations : int, optional
            Maximum number of iterations. If None, runs until evidence termination.
        num_delete : int, default=10
            Number of points to delete per iteration. Higher values improve
            parallelization on GPU.
        num_inner_steps : int, optional
            Number of HRSS steps for each dead point replacement.
            If None, uses 5 * ndim (as recommended).
        termination_threshold : float, default=-3.0
            Terminate when logZ_live - logZ < threshold.
            A value of -3.0 corresponds to about 95% of evidence accumulated.
        rng_key : jax.random.PRNGKey, optional
            Random key. If None, uses PRNGKey(seed).
        seed : int, default=0
            Random seed used if rng_key is None.
        progress : bool, default=True
            Show progress bar.

        Returns
        -------
        dict
            Results dictionary with keys:
            - 'samples': posterior samples (N, ndim)
            - 'loglikelihood': log-likelihood values
            - 'logZ': log-evidence estimate
            - 'logZ_err': log-evidence error estimate
            - 'names': parameter names
            - 'labels': LaTeX labels for parameters
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(seed)

        if num_inner_steps is None:
            num_inner_steps = self.ndim * 5

        # Initialize nested sampler using top-level API
        algo = blackjax.nss(
            logprior_fn=self.log_prior_fn,
            loglikelihood_fn=self.loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )

        # Sample initial live points from prior transform
        rng_key, init_key = jax.random.split(rng_key)
        # Sample from unit cube
        unit_samples = jax.random.uniform(init_key, (n_live, self.ndim))
        # Transform to physical space
        initial_particles = jax.vmap(self.prior_transform)(unit_samples)

        # Initialize state
        live = algo.init(initial_particles)

        # JIT compile the step function
        step_fn = jax.jit(algo.step)

        # Run sampling
        dead = []

        if progress:
            try:
                import tqdm
                pbar = tqdm.tqdm(desc="NS iterations", unit=" iter")
            except ImportError:
                pbar = None
                warnings.warn("tqdm not installed, progress bar disabled")
        else:
            pbar = None

        n_iter = 0
        try:
            while True:
                # Check termination based on evidence
                logZ_diff = live.logZ_live - live.logZ
                if not jnp.isnan(logZ_diff):
                    if logZ_diff < termination_threshold:
                        if pbar is not None:
                            pbar.set_postfix_str(f"Converged: logZ_diff={logZ_diff:.4f}")
                        break

                # Check max iterations
                if max_iterations is not None and n_iter >= max_iterations:
                    if pbar is not None:
                        pbar.set_postfix_str(f"Max iterations reached")
                    break

                # Take step
                rng_key, subkey = jax.random.split(rng_key, 2)
                live, dead_info = step_fn(subkey, live)
                dead.append(dead_info)
                n_iter += 1

                if pbar is not None:
                    pbar.set_postfix(
                        logZ=f'{float(live.logZ):.2f}',
                        diff=f'{float(logZ_diff):.3f}'
                    )
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        # Store dead points for later analysis
        self._dead = dead

        # Finalize results - combine live and dead points
        nested_samples = ns_utils.finalise(live, dead)

        # Extract samples as numpy arrays
        samples = np.asarray(nested_samples.particles)
        loglikelihoods = np.asarray(nested_samples.loglikelihood)

        # Compute log evidence error estimate using log weights
        rng_key, weight_key = jax.random.split(rng_key)
        logw = ns_utils.log_weights(weight_key, nested_samples)
        logZs = jax.scipy.special.logsumexp(logw, axis=0)
        logZ_mean = float(jnp.mean(logZs))
        logZ_err = float(jnp.std(logZs))

        # Store results
        self.results = {
            'samples': samples,
            'loglikelihood': loglikelihoods,
            'logZ': logZ_mean,
            'logZ_err': logZ_err,
            'logZ_runtime': float(live.logZ),  # Evidence from NS run
            'names': self.sampled_names,
            'labels': self.sampled_names_latex,
            'n_iterations': n_iter,
            'n_live': n_live,
            'nested_samples': nested_samples,
        }

        return self.results

    def return_sampled_samples(self, *, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return posterior samples for sampled parameters only.

        Parameters
        ----------
        results : dict, optional
            Results dict from run_sampler(). If None, uses stored results.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'chain': posterior samples (N, ndim)
            - 'names': parameter names
            - 'labels': LaTeX labels
        """
        res = results if results is not None else self.results
        if res is None:
            raise RuntimeError("No results available. Run `run_sampler()` first.")

        return {
            'chain': res['samples'],
            'names': res['names'],
            'labels': res['labels'],
        }

    def return_all_samples(self, *, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return samples for all parameters (sampled + fixed), in original prior order.
        Fixed parameters are filled with their constant values.

        Parameters
        ----------
        results : dict, optional
            Results dict from run_sampler(). If None, uses stored results.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'chain': samples (N, ndim_total) including fixed parameters
            - 'names': all parameter names in original order
            - 'labels': LaTeX labels for all parameters
        """
        res = results if results is not None else self.results
        if res is None:
            raise RuntimeError("No results available. Run `run_sampler()` first.")

        samples = res['samples']
        ns = samples.shape[0]
        chain_all = np.zeros((ns, len(self.discovery_paramnames)), dtype=float)

        # Fill sampled parameters
        for j, name in enumerate(self.sampled_names):
            idx = self.discovery_paramnames.index(name)
            chain_all[:, idx] = samples[:, j]

        # Fill fixed parameters with constant values
        for name in self.fixed_names:
            idx = self.discovery_paramnames.index(name)
            chain_all[:, idx] = float(self.fixed_params[name])

        return {
            'chain': chain_all,
            'names': self.discovery_paramnames,
            'labels': self.latex_list,
        }

    def return_logZ(self, *, results: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
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

        return {
            'logZ': res['logZ'],
            'logZ_err': res['logZ_err'],
        }

    def plot_corner(self, *, burn: int = 0, results: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Create corner plot of posterior samples.

        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        results : dict, optional
            Results dict from run_sampler(). If None, uses stored results.
        **kwargs
            Additional arguments passed to corner.corner()

        Returns
        -------
        matplotlib.figure.Figure
            Corner plot figure
        """
        from .plots import plot_corner

        data = self.return_sampled_samples(results=results)
        return plot_corner(data, burn=burn, **kwargs)

    def plot_trace(self, *, burn: int = 0, plot_fixed: bool = False,
                   results: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Plot trace of samples vs sample index.

        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        plot_fixed : bool, optional
            If True, includes fixed parameters in the plot, by default False.
        results : dict, optional
            Results dict from run_sampler(). If None, uses stored results.
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


__all__ = [
    "DiscoveryBlackJAXBridge",
    "_make_prior_transform_blackjax",
    "_make_log_prior_blackjax",
]
