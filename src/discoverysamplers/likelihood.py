"""
Discovery model adapter utilities.

This module provides adapters for wrapping Discovery models to work with
various sampler backends, including support for JAX JIT compilation and
vectorized evaluation.
"""

import warnings
from typing import Dict, List, Optional


class LikelihoodWrapper:
    """
    Adapter to wrap Discovery models for sampler interfaces.

    This class provides a consistent interface for Discovery models,
    handling JIT compilation and optional vectorization.

    Parameters
    ----------
    model : callable
        Discovery model callable (typically likelihood.logL)
    fixed_params : dict, optional
        Dictionary of fixed parameter values to inject
    jit : bool, default=True
        Whether to JIT-compile the model using JAX
    allow_array_api : bool, default=False
        Whether to support vectorized (batched) evaluation

    Attributes
    ----------
    model : callable
        The wrapped model
    fixed_params : dict
        Fixed parameters to inject
    jit_enabled : bool
        Whether JIT compilation is enabled
    array_api_enabled : bool
        Whether array API is enabled
    array_order : list, optional
        Parameter order for array API

    Examples
    --------
    >>> import discovery as ds
    >>> psr = ds.Pulsar.read_feather('pulsar.feather')
    >>> likelihood = ds.PulsarLikelihood([...])
    >>>
    >>> # Wrap the likelihood
    >>> adapter = _DiscoveryAdapter(
    ...     model=likelihood.logL,
    ...     fixed_params={'param1': 1.0},
    ...     jit=True
    ... )
    >>>
    >>> # Evaluate (fixed params auto-injected)
    >>> log_L = adapter.log_likelihood({'param2': 2.0})
    """

    def __init__(self, model, fixed_params=None, jit=True, allow_array_api=False):
        """Initialize the discovery adapter."""
        self.model = model
        self.logL = model.logL if hasattr(model, 'logL') else model
        self.fixed_params = fixed_params or {}
        self.jit_enabled = jit
        self.allow_array_api = allow_array_api
        self.array_api_enabled = False
        self.array_order = None

        # JIT compile if requested
        if self.jit_enabled:
            self._resolve_jit()

    def _resolve_jit(self):
        """Apply JAX JIT compilation to the model if enabled."""
        if not self.jit_enabled:
            return

        try:
            from jax import jit
            self.logL = jit(self.logL)
        except ImportError:
            warnings.warn(
                "JAX not available, JIT compilation disabled",
                RuntimeWarning
            )
            self.jit_enabled = False

    def configure_array_api(self, order: List[str]):
        """
        Configure vectorized (batched) likelihood evaluation.

        Parameters
        ----------
        order : list of str
            Order of parameters for array construction

        Raises
        ------
        RuntimeError
            If array API was not enabled in __init__

        Examples
        --------
        >>> adapter = _DiscoveryAdapter(model, allow_array_api=True)
        >>> adapter.configure_array_api(['param1', 'param2'])
        >>> # Now can evaluate with batched parameters
        """
        if not self.allow_array_api:
            raise RuntimeError(
                "Array API not enabled. Set allow_array_api=True in __init__"
            )

        self.array_api_enabled = True
        self.array_order = order

    def log_likelihood(self, params_dict: Dict[str, float]) -> float:
        """
        Evaluate log-likelihood for a single parameter set.

        Parameters
        ----------
        params_dict : dict
            Dictionary of parameter values (sampled parameters only)

        Returns
        -------
        float
            Log-likelihood value

        Examples
        --------
        >>> log_L = adapter.log_likelihood({'param2': 2.0, 'param3': 3.0})
        """
        # Merge with fixed parameters
        full_params = {**params_dict, **self.fixed_params}
        return self.logL(full_params)

    def log_likelihood_row(self, params_dict: Dict[str, 'array']):
        """
        Evaluate log-likelihood for a batch of parameter sets (array API).

        Assumes params_dict values are arrays of shape (N,).

        Parameters
        ----------
        params_dict : dict
            Dictionary with array values for each parameter

        Returns
        -------
        array
            Log-likelihood values, shape (N,)

        Raises
        ------
        RuntimeError
            If array API was not configured

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> params = {
        ...     'param1': jnp.array([1.0, 2.0, 3.0]),
        ...     'param2': jnp.array([0.5, 1.0, 1.5])
        ... }
        >>> log_L_batch = adapter.log_likelihood_row(params)
        >>> # Returns array of shape (3,)
        """
        if not self.array_api_enabled:
            raise RuntimeError(
                "Array API not configured. Call configure_array_api() first"
            )

        # Merge with fixed parameters (broadcast to batch)
        full_params = {**params_dict, **self.fixed_params}
        return self.logL(full_params)

    def log_likelihood_matrix(self, params_array):
        """
        Evaluate log-likelihood for a 2D array of parameters.

        Parameters
        ----------
        params_array : array, shape (N, ndim)
            Parameter values for N samples

        Returns
        -------
        array, shape (N,)
            Log-likelihood values

        Raises
        ------
        RuntimeError
            If array API was not configured

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> # Array with 3 samples, 2 parameters each
        >>> params = jnp.array([
        ...     [1.0, 0.5],
        ...     [2.0, 1.0],
        ...     [3.0, 1.5]
        ... ])
        >>> log_L_batch = adapter.log_likelihood_matrix(params)
        >>> # Returns array of shape (3,)
        """
        if not self.array_api_enabled or self.array_order is None:
            raise RuntimeError(
                "Array API not configured. Call configure_array_api() first"
            )

        # Convert array to dict
        params_dict = {
            name: params_array[:, i]
            for i, name in enumerate(self.array_order)
        }

        return self.log_likelihood_row(params_dict)
