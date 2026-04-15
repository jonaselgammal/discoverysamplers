"""
Reversible-jump bridge for Discovery models using Eryn.

This module provides an interface between Discovery models with variable dimensions
(e.g., variable number of gravitational wave sources) and Eryn's reversible-jump
MCMC sampler.

The key components are:
- ``RJ_Discovery_model``: A wrapper that caches likelihoods for all model configurations
  (e.g., 1 source, 2 sources, etc.) and provides a unified ``logL`` function that
  Eryn's RJ sampler can call.  Supports both *variable* branches (RJ) and *fixed*
  branches (always-present parameter groups like pulsar noise or GW background).
- ``DiscoveryErynRJBridge``: The interface class that sets up the Eryn sampler with
  proper priors and handles sampling, result extraction, and plotting.

Example usage (single-branch, backward compatible)::

    rj_model = RJ_Discovery_model(
        psrs=pulsars,
        fixed_components={'per_psr': {'base': make_fixed_components}},
        variable_components={'global': {'cw': (signal_constructor, base_param_names)}},
        variable_component_numbers={'cw': (1, 4)},
    )
    priors = {"cw": {0: uniform_dist(-20, -11), ...}}
    bridge = DiscoveryErynRJBridge(rj_model, priors=priors)
    bridge.create_sampler(nwalkers=32, ntemps=2)
    bridge.run_sampler(nsteps=5000)

Example usage (multi-branch with fixed noise/GW branches)::

    rj_model = RJ_Discovery_model(
        psrs=pulsars,
        fixed_components={'per_psr': {'base': make_fixed_components}},
        variable_components={'global': {'cw': (signal_constructor, cw_param_names)}},
        variable_component_numbers={'cw': (0, 4)},
        fixed_branches={
            'psrn': ['red_noise', 'dm_gp', 'chrom', 'dmexp'],
            'gw': ['gw_'],
        },
    )
    priors = {
        "psrn": {0: ..., 1: ..., ...},
        "gw": {0: ..., 1: ...},
        "cw": {0: ..., 1: ..., ...},
    }
    bridge = DiscoveryErynRJBridge(rj_model, priors=priors)
    bridge.create_sampler(nwalkers=32, ntemps=2)
    bridge.run_sampler(nsteps=5000)

Example usage (single-likelihood mode — one JIT'd likelihood for all configs)::

    rj_model = RJ_Discovery_model(
        psrs=pulsars,
        fixed_components={
            'per_psr': {'base': make_fixed_components},
            'global': {'globalgp': globalgp},
        },
        variable_components={'global': {'cw': (signal_constructor, cw_param_names)}},
        variable_component_numbers={'cw': (0, 5)},
        fixed_branches={
            'psrn': ['red_noise', 'dm_gp', 'chrom', 'dmexp'],
            'gw': ['gw_'],
        },
        single_likelihood=True,  # Build 1 likelihood, silence dead sources
    )

Tested with:
  - discovery >= 0.5
  - eryn >= 1.2
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import warnings
import numpy as np

try:
    from eryn.prior import uniform_dist, ProbDistContainer
    from eryn.ensemble import EnsembleSampler
    from eryn.state import State
    from eryn.moves import GaussianMove
    from eryn.backends import HDFBackend
except ImportError:
    raise ImportError("eryn is not installed. Please install it to use this module.")


class RJ_Discovery_model:
    """
    Discovery model wrapper for reversible-jump MCMC sampling with Eryn.

    This class manages multiple model configurations (e.g., different numbers of
    gravitational wave sources) by pre-computing and caching the likelihood
    for each configuration. It provides a unified ``logL`` interface that Eryn's
    RJ sampler can call with nested parameter lists.

    It also supports **fixed branches**: groups of always-present parameters
    (e.g., pulsar noise, GW background) that are exposed as separate Eryn
    branches with ``nleaves=1``.  This enables Gibbs-style block updates
    between noise, background, and signal parameters.

    Parameters
    ----------
    psrs : list
        List of pulsar objects (Discovery Pulsar instances).
    fixed_components : dict
        Components that don't change in number. Structure::

            {
                'per_psr': {'name': constructor_function},
                'global': {'name': constructor_function}  # optional
            }

        Where constructor_function(psr) returns a list of model components.
    variable_components : dict
        Components that can vary in number. Structure::

            {
                'global': {'branch_name': (constructor_func, base_param_names)}
            }

        Where constructor_func() returns (delay_function, param_names) and
        base_param_names is a list like ['log10_h0', 'log10_f0', ...].
    variable_component_numbers : dict
        Min/max counts for each variable component::

            {'branch_name': (min_count, max_count)}

    fixed_branches : dict or None, optional
        Groups of always-present parameters to expose as separate Eryn branches.
        Keys are branch names, values are lists of substring patterns used to
        match Discovery parameter names.  Example::

            {
                'psrn': ['red_noise', 'dm_gp', 'chrom', 'dmexp'],
                'gw': ['gw_'],
            }

        Parameters matching any pattern for a branch are assigned to that branch.
        Parameters not matching any fixed branch or variable component become
        truly fixed (not sampled).  Default ``None`` (no fixed branches —
        backward-compatible behavior).

    custom_logL : callable or None, optional
        Custom log-likelihood function with signature
        ``custom_logL(param_dict, config) -> float``, where ``param_dict`` is
        a flat dictionary of all parameter values and ``config`` is a dict
        mapping variable component names to their active counts (e.g.,
        ``{'cw': 1}``).  When provided, this replaces the standard
        ``GlobalLikelihood.logL`` call.  Useful for phase-marginalized
        likelihoods.  Default ``None``.

    verbose : bool, optional
        Print detailed information during setup. Default False.

    single_likelihood : bool, optional
        If True, build only ONE likelihood with the maximum number of
        sources and silence dead sources by setting their amplitude to
        ``zero_amplitude_value``.  This avoids caching N separate
        likelihoods and ensures the JAX-compiled likelihood function is
        never recompiled during sampling.  Default False.

    zero_amplitude_value : float, optional
        Value assigned to the amplitude parameter of dead sources in
        single_likelihood mode.  Default -300.0 (i.e., h0 = 10^-300 ≈ 0).

    zero_amplitude_param : str, optional
        Substring used to identify the amplitude parameter among the base
        parameter names.  Default ``"log10_h0"``.

    Notes
    -----
    **single_likelihood edge cases:**

    - When all sources are dead (``n_active=0``), all CW amplitudes are set
      to ``zero_amplitude_value`` and the likelihood equals the noise-only
      value.  Verified to match the standard 0-source likelihood to machine
      precision.
    - Dead-source non-amplitude parameters default to ``0.0``.  This is safe
      because ``h0 ~ 0`` makes the CW contribution vanish regardless of
      other parameter values (the signal is linear in amplitude).
    - This mode requires that the signal model is **linear in the amplitude
      parameter** (i.e., setting amplitude to zero produces zero signal).
      All Discovery CW delay models satisfy this.

    Attributes
    ----------
    likelihood_cache : dict
        Cache of Discovery GlobalLikelihood objects for each configuration.
    params : dict
        Dictionary with 'fixed', 'variable', and 'fixed_branches' parameter info.
    fixed_branch_names : list
        Ordered list of fixed branch names (empty if no fixed branches).
    fixed_branch_param_names : dict
        Maps each fixed branch name to its ordered list of Discovery parameter names.

    Examples
    --------
    >>> rj_model = RJ_Discovery_model(
    ...     psrs=pulsars,
    ...     fixed_components={'per_psr': {'base': make_fixed}},
    ...     variable_components={'global': {'cw': (make_cw_signal, param_names)}},
    ...     variable_component_numbers={'cw': (1, 4)},
    ...     fixed_branches={'psrn': ['red_noise', 'dm_gp'], 'gw': ['gw_']},
    ... )
    """

    def __init__(
        self,
        psrs: List[Any],
        fixed_components: Dict[str, Dict[str, Callable]],
        variable_components: Dict[str, Dict[str, Tuple[Callable, List[str]]]],
        variable_component_numbers: Dict[str, Tuple[int, int]],
        fixed_branches: Optional[Dict[str, List[str]]] = None,
        custom_logL: Optional[Callable] = None,
        verbose: bool = False,
        single_likelihood: bool = False,
        zero_amplitude_value: float = -300.0,
        zero_amplitude_param: str = "log10_h0",
    ) -> None:
        # Delayed import to avoid circular dependencies
        try:
            import discovery as ds
            self._ds = ds
        except ImportError:
            raise ImportError("discovery is not installed. Please install it to use RJ_Discovery_model.")

        self.psrs = psrs
        self.verbose = verbose
        self._custom_logL = custom_logL
        self._single_likelihood = single_likelihood
        self._zero_amplitude_value = zero_amplitude_value
        self._zero_amplitude_param = zero_amplitude_param

        # Parse fixed components
        self.fixed_components = fixed_components
        self.fixed_per_psr = fixed_components.get("per_psr", {})
        # Global fixed components are passed as globalgp to GlobalLikelihood
        # Expected format: {'globalgp': callable_returning_globalgp_component}
        self.fixed_global = fixed_components.get("global", None)

        # Parse variable components
        self.variable_components = variable_components
        self.variable_global = variable_components.get("global", None)
        self.variable_per_psr = variable_components.get("per_psr", None)

        if self.variable_global is None:
            raise ValueError(
                "No global variable components provided. At least one variable component "
                "is required for RJMCMC. Use the standard DiscoveryErynBridge for fixed models."
            )
        if self.variable_per_psr is not None:
            raise NotImplementedError("Variable per-pulsar components are not implemented.")

        self.variable_component_numbers = variable_component_numbers

        # Fixed branches configuration
        self._fixed_branches_spec = fixed_branches or {}
        self.fixed_branch_names: List[str] = list(self._fixed_branches_spec.keys())
        self.fixed_branch_param_names: Dict[str, List[str]] = {}

        # Caches
        self.likelihood_cache: Dict[Tuple, Any] = {}
        self.param_dicts_cache: Dict[Tuple, Dict] = {}
        self.param_mappings_cache: Dict[Tuple, List] = {}

        # Store base parameter names for each variable component
        self.base_param_names_variable: Dict[str, List[str]] = {}
        for comp_name, (constructor, base_names) in self.variable_global.items():
            self.base_param_names_variable[comp_name] = base_names

        # Pre-compute configurations
        if self._single_likelihood:
            self._precompute_single_likelihood()
        else:
            self._precompute_configurations()
        self._determine_all_params()

        # In single_likelihood mode, JIT-compile the max-config likelihood once
        if self._single_likelihood:
            self._setup_single_likelihood_jit()

    def _determine_all_params(self) -> None:
        """Determine fixed, variable, and fixed-branch parameter sets."""
        # Get all params from all configurations and find the config with most parameters
        max_params: List[str] = []
        for likelihood in self.likelihood_cache.values():
            params = likelihood.logL.params
            if len(params) > len(max_params):
                max_params = params

        # Identify variable parameters (belonging to RJ components)
        variable_param_set = set()
        for comp_name, (min_count, max_count) in self.variable_component_numbers.items():
            for param in max_params:
                if any(param.startswith(f"{comp_name}{i}_") for i in range(max_count)):
                    variable_param_set.add(param)

        # Non-variable parameters (candidates for fixed branches or truly fixed)
        non_variable_params = [p for p in max_params if p not in variable_param_set]

        # Assign non-variable params to fixed branches (if specified)
        assigned_to_branch = set()
        for branch_name, patterns in self._fixed_branches_spec.items():
            branch_params = []
            for param in non_variable_params:
                if any(pattern in param for pattern in patterns):
                    branch_params.append(param)
                    assigned_to_branch.add(param)
            self.fixed_branch_param_names[branch_name] = branch_params
            if self.verbose:
                print(f"Fixed branch '{branch_name}': {len(branch_params)} parameters")
                for p in branch_params:
                    print(f"  {p}")

        # Truly fixed params: not variable, not in any fixed branch
        # These need default values since they won't be sampled.
        self.fixed_params = [
            p for p in non_variable_params if p not in assigned_to_branch
        ]
        # Default values for truly fixed params (zeros). Users can override
        # via the ``fixed_param_values`` attribute after construction.
        self.fixed_param_values: Dict[str, float] = {p: 0.0 for p in self.fixed_params}

        if self.verbose and self.fixed_params:
            print(f"Truly fixed (not sampled) parameters: {len(self.fixed_params)}")
            for p in self.fixed_params:
                print(f"  {p}")

        # Variable params are the base names for each component type
        self.variable_params: Dict[str, List[str]] = {}
        for comp_name, (constructor, base_names) in self.variable_global.items():
            self.variable_params[comp_name] = base_names

        self.params = {
            "fixed": self.fixed_params,
            "variable": self.variable_params,
            "fixed_branches": self.fixed_branch_param_names,
        }

    def _precompute_configurations(self, recompute: bool = True) -> None:
        """Pre-compute all valid model configurations and their likelihoods."""
        if self.likelihood_cache and not recompute:
            if self.verbose:
                print("Likelihood configurations already pre-computed, skipping.")
            return

        if self.likelihood_cache and recompute:
            if self.verbose:
                print("Warning: Recomputing likelihood configurations.")
            self.likelihood_cache = {}
            self.param_mappings_cache = {}
            self.param_dicts_cache = {}

        # Get all possible counts for each variable component
        component_ranges = []
        component_names = []

        for comp_name, (min_count, max_count) in self.variable_component_numbers.items():
            component_names.append(comp_name)
            component_ranges.append(range(min_count, max_count + 1))

        # Generate all combinations
        for counts in itertools.product(*component_ranges):
            config = dict(zip(component_names, counts))
            config_key = tuple(sorted(config.items()))

            if self.verbose:
                print(f"Building likelihood for configuration: {config}")

            # Build likelihood for this configuration
            likelihood = self._build_likelihood(config)
            self.likelihood_cache[config_key] = likelihood

            # Determine parameter mapping
            param_dict, param_mapping = self._find_param_mapping(config, likelihood)
            self.param_dicts_cache[config_key] = param_dict
            self.param_mappings_cache[config_key] = param_mapping

            if self.verbose:
                print(f"  Parameters: {likelihood.logL.params}")
                print(f"  Total params: {len(likelihood.logL.params)}")

        if self.verbose:
            print(f"Pre-computed {len(self.likelihood_cache)} model configurations")

    def _precompute_single_likelihood(self) -> None:
        """Build only the max-config likelihood (single_likelihood mode).

        Instead of caching one likelihood per configuration, we build a single
        likelihood with all sources at their maximum count.  Dead sources are
        silenced at evaluation time by setting their amplitude to ~0.
        """
        # Build the max-config only
        max_config = {
            comp_name: max_count
            for comp_name, (_, max_count) in self.variable_component_numbers.items()
        }
        max_config_key = tuple(sorted(max_config.items()))

        if self.verbose:
            print(f"[single_likelihood] Building likelihood for max config: {max_config}")

        likelihood = self._build_likelihood(max_config)
        self.likelihood_cache[max_config_key] = likelihood

        param_dict, param_mapping = self._find_param_mapping(max_config, likelihood)
        self.param_dicts_cache[max_config_key] = param_dict
        self.param_mappings_cache[max_config_key] = param_mapping

        self._max_config = max_config
        self._max_config_key = max_config_key

        if self.verbose:
            print(f"[single_likelihood] Parameters: {likelihood.logL.params}")
            print(f"[single_likelihood] Total params: {len(likelihood.logL.params)}")

        # Build default param values for dead sources.
        # For each variable component, store the "zero" parameter values
        # that silence a source (amplitude → 0, other params at safe defaults).
        self._dead_source_defaults: Dict[str, Dict[str, float]] = {}
        for comp_name, (_, base_names) in self.variable_global.items():
            defaults = {}
            for pname in base_names:
                if self._zero_amplitude_param in pname:
                    defaults[pname] = self._zero_amplitude_value
                else:
                    # Safe midpoint defaults for inactive sources
                    defaults[pname] = 0.0
            self._dead_source_defaults[comp_name] = defaults

        if self.verbose:
            print(f"[single_likelihood] Dead-source defaults: {self._dead_source_defaults}")

    def _setup_single_likelihood_jit(self) -> None:
        """JIT-compile the single max-config likelihood."""
        try:
            import jax
        except ImportError:
            if self.verbose:
                print("[single_likelihood] JAX not available, skipping JIT")
            self._jit_logL = None
            return

        max_lkl = self.likelihood_cache[self._max_config_key]
        self._jit_logL = jax.jit(max_lkl.logL)
        if self.verbose:
            print("[single_likelihood] JIT-compiled max-config likelihood")

    def _build_likelihood(self, config: Dict[str, int]) -> Any:
        """Build a Discovery likelihood for the given configuration."""
        ds = self._ds
        pslmodels = []

        for psr in self.psrs:
            model_components = []

            # Add fixed components
            for comp_name, comp_constructor in self.fixed_per_psr.items():
                component = comp_constructor(psr)
                if isinstance(component, list):
                    model_components.extend(component)
                else:
                    model_components.append(component)

            # Add variable components based on current configuration
            for comp_name, count in config.items():
                if count == 0:
                    continue

                constructor_func, base_names = self.variable_global[comp_name]
                base_delay_fn = constructor_func()[0]  # Get the delay function

                # Add 'count' instances of this component
                for i in range(count):
                    source_name = f"{comp_name}{i}"
                    common_params = [f"{source_name}_{param}" for param in base_names]

                    delay_component = ds.makedelay(
                        psr,
                        base_delay_fn,
                        components=None,
                        common=common_params,
                        name=source_name,
                    )
                    model_components.append(delay_component)

            pslmodels.append(ds.PulsarLikelihood(model_components))

        # Build GlobalLikelihood with optional globalgp
        globalgp_kwargs = {}
        if self.fixed_global is not None:
            # The 'globalgp' key should map to a pre-built globalgp component
            # or a callable that returns one
            globalgp_component = self.fixed_global.get("globalgp", None)
            if globalgp_component is not None:
                if callable(globalgp_component):
                    globalgp_component = globalgp_component()
                globalgp_kwargs["globalgp"] = globalgp_component

        return ds.GlobalLikelihood(psls=pslmodels, **globalgp_kwargs)

    def _find_param_mapping(
        self, config: Dict[str, int], lkl: Any
    ) -> Tuple[Dict[str, Any], List]:
        """
        Find the mapping of parameters for a given configuration.
        
        Returns a dict splitting fixed/variable params and a nested list structure
        that matches what Eryn expects.
        """
        params_list = lkl.logL.params
        
        # Determine variable parameters with their naming scheme
        variable_params: Dict[str, Dict[str, List[str]]] = {}
        for comp_name, count in config.items():
            if count == 0:
                continue
            _, base_names = self.variable_global[comp_name]
            variable_params[comp_name] = {}
            for i in range(count):
                source_name = f"{comp_name}{i}"
                common_params = [f"{source_name}_{param}" for param in base_names]
                variable_params[comp_name][source_name] = common_params

        # Fixed params are all others
        fixed_params = params_list.copy()
        for comp_params in variable_params.values():
            for source_params in comp_params.values():
                for param in source_params:
                    if param in fixed_params:
                        fixed_params.remove(param)

        params_dict = {"fixed": fixed_params, "variable": variable_params}

        # Create nested list structure for Eryn
        param_mapping = [params_dict["fixed"]]
        for comp_name, sources in params_dict["variable"].items():
            comp_list = []
            for i in range(config[comp_name]):
                source_name = f"{comp_name}{i}"
                if source_name in sources:
                    comp_list.append(sources[source_name])
                else:
                    raise ValueError(f"Source name {source_name} not found in variable parameters.")
            param_mapping.append(comp_list)

        return params_dict, param_mapping

    def get_likelihood_for_config(self, config: Dict[str, int]) -> Any:
        """Get the cached likelihood for a given configuration."""
        config_key = tuple(sorted(config.items()))
        return self.likelihood_cache.get(config_key)

    def get_current_config_from_params(self, params: Dict[str, Any]) -> Dict[str, int]:
        """Determine the current model configuration from the parameter dictionary."""
        config = {}
        for comp_name in self.variable_global.keys():
            count = 0
            base_names = self.variable_params[comp_name]
            while True:
                source_name = f"{comp_name}{count}"
                param_exists = any(f"{source_name}_{param}" in params for param in base_names)
                if not param_exists:
                    break
                count += 1
            config[comp_name] = count
        return config

    def _logL(self, params: Dict[str, Any]) -> float:
        """Evaluate log-likelihood for given parameters (dict interface)."""
        config = self.get_current_config_from_params(params)
        likelihood = self.get_likelihood_for_config(config)
        if likelihood is None:
            raise ValueError(f"No likelihood found for configuration: {config}")
        return likelihood.logL(params)

    # ------------------------------------------------------------------
    # Private helpers for logL (split out for readability)
    # ------------------------------------------------------------------

    def _unpack_multi_branch_params(self, params, param_dict):
        """Unpack parameters in multi-branch mode (fixed + variable branches).

        Eryn sends ``[x_branch0, x_branch1, ...]`` as a single list arg when
        ``nbranches > 1``.  Fixed branches are unpacked first, then variable
        (RJ) branches.  *param_dict* is modified in-place.

        Returns
        -------
        dict
            Configuration override mapping each variable component name to its
            number of active sources, derived directly from the array shapes.
        """
        all_branch_data = (
            params[0]
            if len(params) == 1 and isinstance(params[0], list)
            else list(params)
        )

        n_fixed = len(self.fixed_branch_names)

        # Fixed branches: each has shape (1, ndim_branch) — always 1 leaf
        for idx, branch_name in enumerate(self.fixed_branch_names):
            branch_array = all_branch_data[idx]  # shape (1, ndim) or (ndim,)
            if branch_array.ndim == 1:
                branch_array = branch_array[np.newaxis, :]
            branch_param_names = self.fixed_branch_param_names[branch_name]
            for j, pname in enumerate(branch_param_names):
                param_dict[pname] = branch_array[0, j]

        # Variable (RJ) branches: each has shape (n_active_leaves, ndim_branch)
        config_override = {}
        for comp_index, (comp_name, base_names) in enumerate(
            self.variable_params.items()
        ):
            branch_array = all_branch_data[n_fixed + comp_index]
            if branch_array.ndim == 1:
                branch_array = branch_array[np.newaxis, :]
            n_sources = branch_array.shape[0]
            config_override[comp_name] = n_sources
            for source_index in range(n_sources):
                source_name = f"{comp_name}{source_index}"
                for j, param_name in enumerate(base_names):
                    param_dict[f"{source_name}_{param_name}"] = branch_array[
                        source_index, j
                    ]

        return config_override

    def _unpack_legacy_params(self, params, param_dict):
        """Unpack parameters in legacy single-branch mode (backward compatible).

        Handles the original calling convention where fixed parameters come
        first (if any), followed by one array per variable component.
        *param_dict* is modified in-place.

        Returns
        -------
        None
            Always returns ``None`` (no config override in legacy mode).
        """
        if not self.fixed_params:
            fixed_params_arr = []
            offset = 0
        else:
            fixed_params_arr = params[0]
            offset = 1

        for i, param_name in enumerate(self.fixed_params):
            if i < len(fixed_params_arr):
                param_dict[param_name] = fixed_params_arr[i]
            else:
                raise ValueError(
                    f"Not enough fixed parameters. Expected {len(self.fixed_params)}, "
                    f"got {len(fixed_params_arr)}."
                )

        # Variable parameters (subsequent lists)
        for comp_index, (comp_name, base_names) in enumerate(
            self.variable_params.items()
        ):
            comp_param_lists = params[comp_index + offset]
            for source_index, source_params in enumerate(comp_param_lists):
                source_name = f"{comp_name}{source_index}"
                for j, param_name in enumerate(base_names):
                    if j < len(source_params):
                        full_param_name = f"{source_name}_{param_name}"
                        param_dict[full_param_name] = source_params[j]
                    else:
                        raise ValueError(
                            f"Not enough parameters for {source_name}. "
                            f"Expected {len(base_names)}, got {len(source_params)}."
                        )

        return None

    def _pad_dead_sources(self, param_dict, config):
        """Pad dead sources with zero-amplitude defaults (single-likelihood mode).

        Ensures *param_dict* always contains keys for the maximum number of
        sources so that JAX JIT traces remain stable.  Also fills any
        per-pulsar parameters (e.g. ``phi_psr``, ``d_psr``) that are missing.
        *param_dict* is modified in-place.
        """
        for comp_name, (_, max_count) in self.variable_component_numbers.items():
            n_active = config.get(comp_name, 0)
            defaults = self._dead_source_defaults[comp_name]
            base_names = self.variable_params[comp_name]
            for i in range(n_active, max_count):
                source_name = f"{comp_name}{i}"
                for pname in base_names:
                    param_dict[f"{source_name}_{pname}"] = defaults[pname]

        # Also fill per-pulsar params for dead sources (e.g. phi_psr, d_psr)
        max_lkl = self.likelihood_cache[self._max_config_key]
        for p in max_lkl.logL.params:
            if p not in param_dict:
                param_dict[p] = self.fixed_param_values.get(p, 0.0)

    def _evaluate_likelihood(self, param_dict, config, config_override):
        """Dispatch to the appropriate likelihood evaluation path.

        Handles custom log-likelihood, JIT-compiled single-likelihood,
        config-based cache lookup, and the legacy ``_logL`` fallback.

        Parameters
        ----------
        param_dict : dict
            Flat parameter dictionary.
        config : dict
            Current model configuration (component name -> active count).
        config_override : dict or None
            Non-``None`` when the configuration was determined directly from
            multi-branch array shapes (as opposed to inferred from param names).

        Returns
        -------
        float
            The log-likelihood value, or ``-np.inf`` if no cached likelihood
            matches the requested configuration.
        """
        if self._single_likelihood:
            if self._custom_logL is not None:
                return self._custom_logL(param_dict, config)
            elif self._jit_logL is not None:
                return self._jit_logL(param_dict)
            else:
                max_lkl = self.likelihood_cache[self._max_config_key]
                return max_lkl.logL(param_dict)
        elif self._custom_logL is not None:
            return self._custom_logL(param_dict, config)
        elif config_override is not None:
            likelihood = self.get_likelihood_for_config(config_override)
            if likelihood is None:
                return -np.inf
            return likelihood.logL(param_dict)
        else:
            return self._logL(param_dict)

    # ------------------------------------------------------------------
    # Main log-likelihood entry point
    # ------------------------------------------------------------------

    def logL(self, *params) -> float:
        """
        Log-likelihood function for Eryn's RJ sampler.

        When called with a single branch (no fixed branches), Eryn passes
        a single 2D array of shape ``(n_active_leaves, ndim)``.

        When called with multiple branches (fixed + variable), Eryn passes
        a list of 2D arrays, one per branch in the order given by
        ``branch_names`` (fixed branches first, then variable branches).

        Parameters
        ----------
        *params : arrays
            Nested structure of parameters from Eryn.

        Returns
        -------
        float
            Log-likelihood value (or -inf for invalid configurations).
        """
        param_dict = dict(self.fixed_param_values)

        if self.fixed_branch_names:
            config_override = self._unpack_multi_branch_params(params, param_dict)
        else:
            config_override = self._unpack_legacy_params(params, param_dict)

        config = (
            config_override
            if config_override is not None
            else self.get_current_config_from_params(param_dict)
        )

        if self._single_likelihood:
            self._pad_dead_sources(param_dict, config)

        logL_val = self._evaluate_likelihood(param_dict, config, config_override)
        return float(
            np.nan_to_num(logL_val, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        )

    def get_all_configurations(self) -> List[Dict[str, int]]:
        """Return all pre-computed configurations."""
        return [dict(config_key) for config_key in self.likelihood_cache.keys()]

    def params_all_configurations(self) -> List[str]:
        """Return the list of all parameters across all configurations."""
        all_params: set = set()
        for likelihood in self.likelihood_cache.values():
            all_params.update(likelihood.logL.params)
        return sorted(all_params)

    def get_param_dict_for_config(self, config: Dict[str, int]) -> Optional[Dict]:
        """Get parameter dictionary for a specific configuration."""
        return self.param_dicts_cache.get(tuple(sorted(config.items())))

    def get_param_mapping_for_config(self, config: Dict[str, int]) -> Optional[List]:
        """Get parameter mapping for a specific configuration."""
        return self.param_mappings_cache.get(tuple(sorted(config.items())))


class DiscoveryErynRJBridge:
    """
    Bridge between RJ_Discovery_model and Eryn's reversible-jump MCMC sampler.

    Supports two modes:

    1. **Single-branch** (backward compatible): only variable (RJ) branches.
    2. **Multi-branch**: fixed branches (always-present, ``nleaves=1``) alongside
       variable (RJ) branches, enabling Gibbs-style block updates.

    Parameters
    ----------
    rj_model : RJ_Discovery_model
        The model with cached likelihoods for all configurations.
    priors : dict
        Priors in Eryn RJ format::

            {
                "branch_name": {
                    0: prior_for_param_0,
                    1: prior_for_param_1,
                    ...
                }
            }

        Must include entries for every fixed branch (if any) **and** every
        variable branch.
    latex_labels : dict, optional
        LaTeX labels for parameter names.

    Attributes
    ----------
    all_branch_names : list
        All branch names (fixed + variable), in the order Eryn sees them.
    rj_branch_names : list
        Names of variable (RJ) branches only.
    fixed_branch_names : list
        Names of fixed (always-present) branches only.
    ndims_dict : dict
        Maps branch name -> number of parameters in that branch.
    nleaves_min_dict / nleaves_max_dict : dict
        Maps branch name -> min/max leaf count.
    """

    def __init__(
        self,
        rj_model: RJ_Discovery_model,
        priors: Dict[str, Dict[int, Any]],
        latex_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self.rj_model = rj_model
        self.priors = priors
        self.latex_labels = latex_labels or {}

        # ---- Identify branches ----
        self.fixed_branch_names = list(rj_model.fixed_branch_names)
        self.rj_branch_names = list(rj_model.variable_component_numbers.keys())
        # Eryn branch order: fixed branches first, then variable
        self.all_branch_names = self.fixed_branch_names + self.rj_branch_names

        # ---- Dimensions per branch ----
        self.ndims_dict: Dict[str, int] = {}
        for branch in self.fixed_branch_names:
            self.ndims_dict[branch] = len(rj_model.fixed_branch_param_names[branch])
        for branch in self.rj_branch_names:
            self.ndims_dict[branch] = len(rj_model.variable_params[branch])

        # ---- Leaf counts per branch ----
        self.nleaves_min_dict: Dict[str, int] = {}
        self.nleaves_max_dict: Dict[str, int] = {}
        for branch in self.fixed_branch_names:
            self.nleaves_min_dict[branch] = 1
            self.nleaves_max_dict[branch] = 1
        for branch in self.rj_branch_names:
            mn, mx = rj_model.variable_component_numbers[branch]
            self.nleaves_min_dict[branch] = mn
            self.nleaves_max_dict[branch] = mx

        # ---- Backward-compatible shortcuts for single RJ branch ----
        if len(self.rj_branch_names) == 1:
            first_rj = self.rj_branch_names[0]
            self.ndim = self.ndims_dict[first_rj]
            self.nleaves_min = self.nleaves_min_dict[first_rj]
            self.nleaves_max = self.nleaves_max_dict[first_rj]
            self.base_param_names = rj_model.variable_params[first_rj]
            self.latex_list = [
                self.latex_labels.get(name, name) for name in self.base_param_names
            ]
        else:
            # Multiple RJ branches — use first for backward compat attrs
            first_rj = self.rj_branch_names[0]
            self.ndim = self.ndims_dict[first_rj]
            self.nleaves_min = self.nleaves_min_dict[first_rj]
            self.nleaves_max = self.nleaves_max_dict[first_rj]
            self.base_param_names = rj_model.variable_params[first_rj]
            self.latex_list = [
                self.latex_labels.get(name, name) for name in self.base_param_names
            ]

        # Also keep the old attribute name for backward compatibility
        self.branch_names = self.all_branch_names

        # Store param name lists for each branch (for results extraction)
        self._branch_param_names: Dict[str, List[str]] = {}
        for branch in self.fixed_branch_names:
            self._branch_param_names[branch] = rj_model.fixed_branch_param_names[branch]
        for branch in self.rj_branch_names:
            self._branch_param_names[branch] = rj_model.variable_params[branch]

        # Sampler will be created later
        self.sampler: Optional[EnsembleSampler] = None
        self.nwalkers: Optional[int] = None
        self.ntemps: int = 1

    @property
    def has_fixed_branches(self) -> bool:
        """Whether this bridge has fixed (always-present) branches."""
        return len(self.fixed_branch_names) > 0

    def create_sampler(
        self,
        nwalkers: int,
        ntemps: int = 1,
        moves: Optional[Any] = None,
        move_cov_factor: float = 0.01,
        rj_moves: Any = True,
        checkpoint_file: Optional[str] = None,
        **kwargs,
    ) -> EnsembleSampler:
        """
        Create the Eryn ensemble sampler for RJMCMC.

        Parameters
        ----------
        nwalkers : int
            Number of walkers per temperature.
        ntemps : int, optional
            Number of temperatures for parallel tempering. Default 1.
        moves : eryn Move, optional
            Custom move proposal. If None, uses GaussianMove with diagonal covariance.
        move_cov_factor : float, optional
            Factor for diagonal covariance in default GaussianMove. Default 0.01.
        rj_moves : bool or str or list, optional
            Reversible-jump move configuration passed to Eryn. For multi-branch
            setups, ``"separate_branches"`` is recommended. Default ``True``.
        checkpoint_file : str, optional
            Path to an HDF5 file for checkpointing. If provided, the sampler
            stores all chain data to this file and can be resumed from it.
            If the file already exists and contains data, the sampler will
            resume from the last saved state when ``run_sampler`` is called
            with ``initial_state=None``.
        **kwargs
            Additional arguments passed to EnsembleSampler.

        Returns
        -------
        EnsembleSampler
            The configured Eryn sampler.
        """
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self._checkpoint_file = checkpoint_file

        # Set up HDF5 backend for checkpointing
        if checkpoint_file is not None and "backend" not in kwargs:
            kwargs["backend"] = HDFBackend(checkpoint_file)

        # Build ndims, nleaves_min, nleaves_max as dicts for Eryn
        ndims = {b: self.ndims_dict[b] for b in self.all_branch_names}
        nleaves_max = {b: self.nleaves_max_dict[b] for b in self.all_branch_names}
        nleaves_min = {b: self.nleaves_min_dict[b] for b in self.all_branch_names}

        # Default moves: Gaussian with diagonal covariance per branch
        if moves is None:
            cov = {
                branch: np.diag(np.ones(self.ndims_dict[branch])) * move_cov_factor
                for branch in self.all_branch_names
            }
            moves = GaussianMove(cov)

        # Tempering kwargs
        tempering_kwargs = kwargs.pop("tempering_kwargs", None)
        if tempering_kwargs is None and ntemps > 1:
            tempering_kwargs = {"ntemps": ntemps}

        # For multi-branch with fixed branches, recommend "separate_branches"
        if self.has_fixed_branches and rj_moves is True:
            rj_moves = "separate_branches"

        self.sampler = EnsembleSampler(
            nwalkers,
            ndims,
            self.rj_model.logL,
            priors=self.priors,
            tempering_kwargs=tempering_kwargs,
            nbranches=len(self.all_branch_names),
            branch_names=self.all_branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves=rj_moves,
            **kwargs,
        )

        return self.sampler

    def initialize_state(
        self,
        initial_nleaves: Optional[Union[int, Dict[str, int]]] = None,
        initial_points: Optional[Dict[str, np.ndarray]] = None,
        scatter: float = 1e-6,
    ) -> State:
        """
        Initialize the sampler state.

        Parameters
        ----------
        initial_nleaves : int or dict, optional
            Number of active leaves to start with.  If ``int``, applies to
            the first RJ branch. If ``dict``, maps branch name -> count.
            Fixed branches always start with 1 leaf. Defaults to nleaves_min
            for RJ branches.
        initial_points : dict, optional
            Initial parameter values per branch.  Keys are branch names,
            values are arrays of shape ``(nleaves, ndim)``.  If ``None``,
            draws from priors.
        scatter : float, optional
            Standard deviation for Gaussian scatter around initial points.
            Default 1e-6.

        Returns
        -------
        State
            Eryn State object ready for sampling.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        # Normalize initial_nleaves to dict
        if initial_nleaves is None:
            initial_nleaves_dict = {}
        elif isinstance(initial_nleaves, int):
            # Apply to first RJ branch only
            initial_nleaves_dict = {self.rj_branch_names[0]: initial_nleaves}
        else:
            initial_nleaves_dict = initial_nleaves

        initial_points = initial_points or {}

        coords = {}
        inds = {}

        for branch in self.all_branch_names:
            ndim_b = self.ndims_dict[branch]
            nleaves_max_b = self.nleaves_max_dict[branch]

            # Determine how many leaves to activate
            if branch in self.fixed_branch_names:
                n_active = 1  # Fixed branches always have 1 leaf
            else:
                n_active = initial_nleaves_dict.get(
                    branch, self.nleaves_min_dict[branch]
                )
                n_active = max(n_active, self.nleaves_min_dict[branch])

            coords[branch] = np.zeros(
                (self.ntemps, self.nwalkers, nleaves_max_b, ndim_b)
            )
            inds[branch] = np.zeros(
                (self.ntemps, self.nwalkers, nleaves_max_b), dtype=bool
            )
            inds[branch][:, :, :n_active] = True

            if branch in initial_points:
                # Use provided initial point with scatter
                init_pt = initial_points[branch]
                for nn in range(min(n_active, nleaves_max_b)):
                    for i in range(ndim_b):
                        pt = init_pt[nn, i] if nn < len(init_pt) else init_pt[0, i]
                        coords[branch][:, :, nn, i] = np.random.normal(
                            loc=pt, scale=scatter,
                            size=(self.ntemps, self.nwalkers),
                        )
            else:
                # Draw from priors
                if branch in self.priors:
                    for nn in range(n_active):
                        for i, prior in self.priors[branch].items():
                            coords[branch][:, :, nn, i] = prior.rvs(
                                size=(self.ntemps, self.nwalkers)
                            )

        # Compute initial log-prior and log-likelihood
        log_prior = self.sampler.compute_log_prior(coords, inds=inds)
        log_like = self.sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)
        return state

    def run_sampler(
        self,
        nsteps: int,
        initial_state: Optional[State] = None,
        initial_nleaves: Optional[Union[int, Dict[str, int]]] = None,
        initial_points: Optional[Dict[str, np.ndarray]] = None,
        burn: int = 0,
        thin_by: int = 1,
        progress: bool = True,
        # Backward compat: accept old kwarg name
        initial_point: Optional[np.ndarray] = None,
        **kwargs,
    ) -> State:
        """
        Run the RJMCMC sampler.

        Parameters
        ----------
        nsteps : int
            Number of MCMC steps to run.
        initial_state : State, optional
            Starting state. If None, creates one using initialize_state().
        initial_nleaves : int or dict, optional
            Passed to initialize_state() if initial_state is None.
        initial_points : dict, optional
            Passed to initialize_state() if initial_state is None.
        initial_point : array, optional
            **Deprecated.** Use ``initial_points`` instead. If provided,
            applied to the first RJ branch.
        burn : int, optional
            Burn-in steps to discard. Default 0.
        thin_by : int, optional
            Thinning factor. Default 1.
        progress : bool, optional
            Show progress bar. Default True.
        **kwargs
            Additional arguments passed to sampler.run_mcmc().

        Returns
        -------
        State
            Final state after sampling.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        # Backward compatibility for initial_point (singular)
        if initial_point is not None:
            warnings.warn(
                "initial_point is deprecated. Use initial_points={'branch_name': array} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if initial_points is None:
                initial_points = {self.rj_branch_names[0]: initial_point}

        if initial_state is None:
            initial_state = self.initialize_state(
                initial_nleaves=initial_nleaves,
                initial_points=initial_points,
            )

        final_state = self.sampler.run_mcmc(
            initial_state,
            nsteps,
            burn=burn,
            thin_by=thin_by,
            progress=progress,
            **kwargs,
        )

        return final_state

    def get_last_state(self) -> State:
        """
        Return the last sampler state.

        This can be passed as ``initial_state`` to ``run_sampler`` to continue
        sampling from where the previous run left off, even without an HDF5
        backend.

        Returns
        -------
        State
            The last sampler state.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")
        return self.sampler.get_last_sample()

    @property
    def checkpoint_file(self) -> Optional[str]:
        """Path to the HDF5 checkpoint file, or None."""
        return self._checkpoint_file

    @property
    def can_resume(self) -> bool:
        """Whether the sampler can resume from a checkpoint file."""
        if self.sampler is None or self._checkpoint_file is None:
            return False
        return self.sampler.backend.initialized

    @property
    def completed_steps(self) -> int:
        """Number of completed steps stored in the backend."""
        if self.sampler is None:
            return 0
        return self.sampler.iteration

    def resume(
        self,
        nsteps: int,
        progress: bool = True,
        **kwargs,
    ) -> State:
        """
        Resume sampling from the last checkpoint.

        Continues from the last stored state in the HDF5 backend (or the
        last in-memory state) for ``nsteps`` additional steps.

        Parameters
        ----------
        nsteps : int
            Number of additional MCMC steps to run.
        progress : bool, optional
            Show progress bar. Default True.
        **kwargs
            Additional arguments passed to sampler.run_mcmc().

        Returns
        -------
        State
            Final state after the additional steps.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        last_state = self.sampler.get_last_sample()

        final_state = self.sampler.run_mcmc(
            last_state,
            nsteps,
            progress=progress,
            **kwargs,
        )
        return final_state

    def return_sampled_samples(
        self, branch: Optional[str] = None, temperature: int = 0
    ) -> Dict[str, Any]:
        """
        Return the sampled parameter chains for a branch.

        Parameters
        ----------
        branch : str, optional
            Branch name. Defaults to first RJ branch.
        temperature : int, optional
            Temperature index. Default 0 (coldest).

        Returns
        -------
        dict
            Dictionary with 'names', 'labels', and 'chain' keys.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        if branch is None:
            branch = self.rj_branch_names[0]

        chain = self.sampler.get_chain()[branch]
        chain_temp = chain[:, temperature]

        param_names = self._branch_param_names[branch]
        labels = [self.latex_labels.get(n, n) for n in param_names]

        return {
            "names": param_names,
            "labels": labels,
            "chain": chain_temp,
        }

    def return_flat_samples(
        self, branch: Optional[str] = None, temperature: int = 0
    ) -> np.ndarray:
        """
        Return flattened samples, excluding inactive (NaN) entries.

        Parameters
        ----------
        branch : str, optional
            Branch name. Defaults to first RJ branch.
        temperature : int, optional
            Temperature index. Default 0.

        Returns
        -------
        ndarray
            Flattened samples of shape (n_valid_samples, ndim).
        """
        samples = self.return_sampled_samples(branch=branch, temperature=temperature)
        chain = samples["chain"]
        ndim_b = chain.shape[-1]

        flat_chain = chain.reshape(-1, ndim_b)
        valid_mask = ~np.isnan(flat_chain[:, 0])

        return flat_chain[valid_mask]

    def return_nleaves(self, branch: Optional[str] = None) -> np.ndarray:
        """
        Return the number of active leaves at each step.

        Parameters
        ----------
        branch : str, optional
            Branch name. Defaults to first RJ branch.

        Returns
        -------
        ndarray
            Array of shape (nsteps, ntemps, nwalkers) with leaf counts.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        if branch is None:
            branch = self.rj_branch_names[0]

        return self.sampler.get_nleaves()[branch]

    def return_logZ(self, *, results=None) -> Dict[str, float]:
        """Not supported for MCMC samplers."""
        raise NotImplementedError(
            "Eryn RJMCMC is an MCMC sampler and does not compute Bayesian evidence (logZ). "
            "Use nested sampling (Nessai, JAX-NS) for evidence estimates."
        )

    def plot_nleaves_histogram(
        self,
        branch: Optional[str] = None,
        figsize: Tuple[int, int] = (6, 12),
    ) -> "plt.Figure":
        """
        Plot histogram of the number of active leaves at each temperature.

        Parameters
        ----------
        branch : str, optional
            RJ branch to plot. Defaults to first RJ branch.
        figsize : tuple, optional
            Figure size. Default (6, 12).

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt

        if branch is None:
            branch = self.rj_branch_names[0]

        nleaves = self.return_nleaves(branch=branch)
        nleaves_min_b = self.nleaves_min_dict[branch]
        nleaves_max_b = self.nleaves_max_dict[branch]

        bins = np.arange(nleaves_min_b - 0.5, nleaves_max_b + 1.5)

        fig, axes = plt.subplots(self.ntemps, 1, sharex=True, figsize=figsize)
        if self.ntemps == 1:
            axes = [axes]

        axes[-1].set_xlabel("Number of Models")
        axes[-1].set_xticks(np.arange(nleaves_min_b, nleaves_max_b + 1))

        for temp, ax in enumerate(axes):
            ax.hist(nleaves[:, temp].flatten(), bins=bins)
            ax.text(
                1.02, 0.45, f"Temperature {temp}",
                horizontalalignment="left",
                transform=ax.transAxes,
            )

        fig.tight_layout()
        return fig

    def plot_corner(
        self, branch: Optional[str] = None, temperature: int = 0, **kwargs
    ) -> "plt.Figure":
        """
        Create a corner plot of the sampled parameters.

        Parameters
        ----------
        branch : str, optional
            Branch name. Defaults to first RJ branch.
        temperature : int, optional
            Temperature index. Default 0.
        **kwargs
            Additional arguments passed to corner.corner().

        Returns
        -------
        matplotlib.figure.Figure
            The corner plot figure.
        """
        import corner

        samples = self.return_sampled_samples(branch=branch, temperature=temperature)
        flat_chain = samples["chain"].reshape(-1, samples["chain"].shape[-1])
        valid = ~np.isnan(flat_chain[:, 0])

        fig = corner.corner(
            flat_chain[valid],
            labels=samples["labels"],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            **kwargs,
        )

        return fig

    def print_config_summary(self) -> None:
        """Print a summary of all model configurations."""
        print("RJ Model Configuration Summary")
        print("=" * 50)
        if self.has_fixed_branches:
            print(f"Fixed branches: {self.fixed_branch_names}")
            for b in self.fixed_branch_names:
                print(f"  {b}: {self.ndims_dict[b]} params (always active)")
        print(f"RJ branches: {self.rj_branch_names}")
        for b in self.rj_branch_names:
            print(
                f"  {b}: {self.ndims_dict[b]} params per source, "
                f"leaves [{self.nleaves_min_dict[b]}, {self.nleaves_max_dict[b]}]"
            )
        print()
        if self.rj_model._single_likelihood:
            print(f"Mode: single_likelihood (1 JIT-compiled likelihood for all configs)")
            max_cfg = self.rj_model._max_config
            max_lkl = self.rj_model.likelihood_cache[self.rj_model._max_config_key]
            print(f"  Max config {max_cfg}: {len(max_lkl.logL.params)} total parameters")
            print(f"  Dead sources silenced via {self.rj_model._zero_amplitude_param} = {self.rj_model._zero_amplitude_value}")
        else:
            print("Available configurations:")
            for config in self.rj_model.get_all_configurations():
                likelihood = self.rj_model.get_likelihood_for_config(config)
                n_params = len(likelihood.logL.params)
                print(f"  {config}: {n_params} total parameters")


# Backwards compatibility: ErynRJBridge was the original class name.
# Use DiscoveryErynRJBridge in new code; ErynRJBridge is kept as an alias.
ErynRJBridge = DiscoveryErynRJBridge

__all__ = ["RJ_Discovery_model", "DiscoveryErynRJBridge", "ErynRJBridge", "GaussianMove"]
