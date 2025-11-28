"""
Reversible-jump bridge for Discovery models using Eryn.

This module provides an interface between Discovery models with variable dimensions
(e.g., variable number of gravitational wave sources) and Eryn's reversible-jump
MCMC sampler.

The key components are:
- ``RJ_Discovery_model``: A wrapper that caches likelihoods for all model configurations
  (e.g., 1 source, 2 sources, etc.) and provides a unified ``logL`` function that
  Eryn's RJ sampler can call.
- ``DiscoveryErynRJBridge``: The interface class that sets up the Eryn sampler with
  proper priors and handles sampling, result extraction, and plotting.

Example usage::

    # 1. Create the RJ model with cached likelihoods
    rj_model = RJ_Discovery_model(
        psrs=pulsars,
        fixed_components={'per_psr': {'base': make_fixed_components}},
        variable_components={'global': {'cw': (signal_constructor, base_param_names)}},
        variable_component_numbers={'cw': (1, 4)},  # 1 to 4 sources
    )

    # 2. Define priors (Eryn format: branch -> index -> distribution)
    priors = {
        "cw": {
            0: uniform_dist(-20, -11),  # log10_h0
            1: uniform_dist(-9, -7),     # log10_f0
            ...
        }
    }

    # 3. Create the bridge and run
    bridge = DiscoveryErynRJBridge(rj_model, priors=priors)
    bridge.create_sampler(nwalkers=32, ntemps=2)
    bridge.run_sampler(nsteps=5000)

Tested with:
  - discovery >= 0.5
  - eryn >= 1.2
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import warnings
import numpy as np

try:
    from eryn.prior import uniform_dist, ProbDistContainer
    from eryn.ensemble import EnsembleSampler
    from eryn.state import State
    from eryn.moves import GaussianMove
except ImportError:
    raise ImportError("eryn is not installed. Please install it to use this module.")


class RJ_Discovery_model:
    """
    Discovery model wrapper for reversible-jump MCMC sampling with Eryn.
    
    This class manages multiple model configurations (e.g., different numbers of
    gravitational wave sources) by pre-computing and caching the likelihood
    for each configuration. It provides a unified ``logL`` interface that Eryn's
    RJ sampler can call with nested parameter lists.

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
            
    verbose : bool, optional
        Print detailed information during setup. Default False.

    Attributes
    ----------
    likelihood_cache : dict
        Cache of Discovery GlobalLikelihood objects for each configuration.
    params : dict
        Dictionary with 'fixed' and 'variable' parameter information.
    
    Examples
    --------
    >>> rj_model = RJ_Discovery_model(
    ...     psrs=pulsars,
    ...     fixed_components={'per_psr': {'base': make_fixed}},
    ...     variable_components={'global': {'cw': (make_cw_signal, param_names)}},
    ...     variable_component_numbers={'cw': (1, 4)},
    ... )
    >>> print(rj_model.get_all_configurations())
    [{'cw': 1}, {'cw': 2}, {'cw': 3}, {'cw': 4}]
    """

    def __init__(
        self,
        psrs: List[Any],
        fixed_components: Dict[str, Dict[str, Callable]],
        variable_components: Dict[str, Dict[str, Tuple[Callable, List[str]]]],
        variable_component_numbers: Dict[str, Tuple[int, int]],
        verbose: bool = False,
    ) -> None:
        # Delayed import to avoid circular dependencies
        try:
            import discovery as ds
            self._ds = ds
        except ImportError:
            raise ImportError("discovery is not installed. Please install it to use RJ_Discovery_model.")

        self.psrs = psrs
        self.verbose = verbose

        # Parse fixed components
        self.fixed_components = fixed_components
        self.fixed_per_psr = fixed_components.get("per_psr", {})
        self.fixed_global = fixed_components.get("global", None)
        if self.fixed_global is not None:
            raise NotImplementedError(
                "Fixed global components are not yet implemented. "
                "Discovery's GlobalLikelihood doesn't support them in the expected way."
            )

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

        # Caches
        self.likelihood_cache: Dict[Tuple, Any] = {}
        self.param_dicts_cache: Dict[Tuple, Dict] = {}
        self.param_mappings_cache: Dict[Tuple, List] = {}

        # Store base parameter names for each variable component
        self.base_param_names_variable: Dict[str, List[str]] = {}
        for comp_name, (constructor, base_names) in self.variable_global.items():
            self.base_param_names_variable[comp_name] = base_names

        # Pre-compute all possible configurations
        self._precompute_configurations()
        self._determine_all_params()

    def _determine_all_params(self) -> None:
        """Determine fixed and variable parameter sets across all configurations."""
        # Get all params from all configurations and find the config with most parameters
        max_params: List[str] = []
        for likelihood in self.likelihood_cache.values():
            params = likelihood.logL.params
            if len(params) > len(max_params):
                max_params = params

        # Extract fixed parameters by removing variable ones
        fixed_params = []
        for param in max_params:
            is_variable = False
            for comp_name, (min_count, max_count) in self.variable_component_numbers.items():
                if any(param.startswith(f"{comp_name}{i}_") for i in range(max_count)):
                    is_variable = True
                    break
            if not is_variable:
                fixed_params.append(param)

        self.fixed_params = fixed_params

        # Variable params are the base names for each component type
        self.variable_params: Dict[str, List[str]] = {}
        for comp_name, (constructor, base_names) in self.variable_global.items():
            self.variable_params[comp_name] = base_names

        self.params = {
            "fixed": self.fixed_params,
            "variable": self.variable_params,
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

        return ds.GlobalLikelihood(psls=pslmodels)

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

    def logL(self, *params) -> float:
        """
        Log-likelihood function for Eryn's RJ sampler.
        
        Eryn passes nested lists of parameters where:
        - First list: fixed parameters (or empty if none)
        - Subsequent lists: variable parameters grouped by branch/component type,
          where each is a list of arrays (one per source)

        Parameters
        ----------
        *params : arrays
            Nested structure of parameters from Eryn.

        Returns
        -------
        float
            Log-likelihood value (or -inf for invalid configurations).
        """
        param_dict = {}

        # Fixed parameters (first list if we have fixed params)
        if not self.fixed_params:
            fixed_params = []
            offset = 0
        else:
            fixed_params = params[0]
            offset = 1
            
        for i, param_name in enumerate(self.fixed_params):
            if i < len(fixed_params):
                param_dict[param_name] = fixed_params[i]
            else:
                raise ValueError(
                    f"Not enough fixed parameters. Expected {len(self.fixed_params)}, got {len(fixed_params)}."
                )

        # Variable parameters (subsequent lists)
        for comp_index, (comp_name, base_names) in enumerate(self.variable_params.items()):
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

        logL_val = self._logL(param_dict)
        return float(np.nan_to_num(logL_val, nan=-np.inf, posinf=np.inf, neginf=-np.inf))

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

    This class provides a complete interface for running RJMCMC on Discovery models
    with variable numbers of signal components. It handles:
    
    - Prior specification in Eryn's format (branch -> parameter index -> distribution)
    - Sampler creation with proper configuration for RJ moves
    - Initial state setup with configurable starting configuration
    - Result extraction and plotting

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
            
        Where each prior is an eryn distribution (e.g., ``uniform_dist(-20, -11)``).
    latex_labels : dict, optional
        LaTeX labels for base parameter names (e.g., ``{'log10_h0': r'$\\log_{10} h_0$'}``).

    Attributes
    ----------
    rj_model : RJ_Discovery_model
        The underlying RJ model with cached likelihoods.
    branch_names : list
        Names of all branches (variable component types).
    ndim : int
        Number of parameters per source (for the first/only variable component).
    nleaves_min : int
        Minimum number of leaves (sources) allowed.
    nleaves_max : int
        Maximum number of leaves (sources) allowed.

    Examples
    --------
    >>> from eryn.prior import uniform_dist
    >>> priors = {
    ...     "cw": {
    ...         0: uniform_dist(-20, -11),  # log10_h0
    ...         1: uniform_dist(-9, -7),     # log10_f0
    ...         2: uniform_dist(0, 2*np.pi), # ra
    ...         3: uniform_dist(-1, 1),      # sindec
    ...         4: uniform_dist(-1, 1),      # cosinc
    ...         5: uniform_dist(0, np.pi),   # psi
    ...         6: uniform_dist(0, 2*np.pi), # phi_earth
    ...         7: uniform_dist(7, 10),      # log10_Mc
    ...     }
    ... }
    >>> bridge = DiscoveryErynRJBridge(rj_model, priors=priors)
    >>> bridge.create_sampler(nwalkers=32, ntemps=2)
    >>> bridge.run_sampler(nsteps=5000, progress=True)
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

        # Extract branch information
        self.branch_names = list(rj_model.variable_component_numbers.keys())
        
        # For now, assume single branch (most common case)
        if len(self.branch_names) != 1:
            warnings.warn(
                f"Multiple branches detected: {self.branch_names}. "
                "This interface is optimized for single-branch RJMCMC."
            )

        # Get dimensions from the first branch
        first_branch = self.branch_names[0]
        self.ndim = len(rj_model.variable_params[first_branch])
        self.nleaves_min, self.nleaves_max = rj_model.variable_component_numbers[first_branch]

        # Store base parameter names for labeling
        self.base_param_names = rj_model.variable_params[first_branch]

        # Create latex labels list
        self.latex_list = [
            self.latex_labels.get(name, name) for name in self.base_param_names
        ]

        # Sampler will be created later
        self.sampler: Optional[EnsembleSampler] = None
        self.nwalkers: Optional[int] = None
        self.ntemps: int = 1

    def create_sampler(
        self,
        nwalkers: int,
        ntemps: int = 1,
        moves: Optional[Any] = None,
        move_cov_factor: float = 0.01,
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
        **kwargs
            Additional arguments passed to EnsembleSampler.

        Returns
        -------
        EnsembleSampler
            The configured Eryn sampler.
        """
        self.nwalkers = nwalkers
        self.ntemps = ntemps

        # Default moves: Gaussian with diagonal covariance
        if moves is None:
            cov = {branch: np.diag(np.ones(self.ndim)) * move_cov_factor 
                   for branch in self.branch_names}
            moves = GaussianMove(cov)

        # Tempering kwargs
        tempering_kwargs = kwargs.pop("tempering_kwargs", None)
        if tempering_kwargs is None and ntemps > 1:
            tempering_kwargs = {"ntemps": ntemps}

        self.sampler = EnsembleSampler(
            nwalkers,
            self.ndim,
            self.rj_model.logL,
            priors=self.priors,
            tempering_kwargs=tempering_kwargs,
            nbranches=len(self.branch_names),
            branch_names=self.branch_names,
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            moves=moves,
            rj_moves=True,  # Enable reversible-jump moves
            **kwargs,
        )

        return self.sampler

    def initialize_state(
        self,
        initial_nleaves: Optional[int] = None,
        initial_point: Optional[np.ndarray] = None,
        scatter: float = 1e-6,
    ) -> State:
        """
        Initialize the sampler state.

        Parameters
        ----------
        initial_nleaves : int, optional
            Number of sources to start with. Defaults to nleaves_min.
        initial_point : array, optional
            Initial parameter values for active sources. Shape (nleaves, ndim).
            If None, draws from priors.
        scatter : float, optional
            Standard deviation for Gaussian scatter around initial_point. Default 1e-6.

        Returns
        -------
        State
            Eryn State object ready for sampling.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        if initial_nleaves is None:
            initial_nleaves = self.nleaves_min

        # Initialize coordinates
        coords = {}
        inds = {}

        for branch in self.branch_names:
            coords[branch] = np.zeros((self.ntemps, self.nwalkers, self.nleaves_max, self.ndim))
            inds[branch] = np.zeros((self.ntemps, self.nwalkers, self.nleaves_max), dtype=bool)

            # Set active leaves
            inds[branch][:, :, :initial_nleaves] = True

            # Fill coordinates
            if initial_point is not None:
                # Use provided initial point with scatter
                for nn in range(min(initial_nleaves, self.nleaves_max)):
                    for i in range(self.ndim):
                        coords[branch][:, :, nn, i] = np.random.normal(
                            loc=initial_point[nn, i] if nn < len(initial_point) else initial_point[0, i],
                            scale=scatter,
                            size=(self.ntemps, self.nwalkers),
                        )
            else:
                # Draw from priors
                for nn in range(initial_nleaves):
                    for i, prior in self.priors[branch].items():
                        coords[branch][:, :, nn, i] = prior.rvs(size=(self.ntemps, self.nwalkers))

        # Compute initial log-prior and log-likelihood
        log_prior = self.sampler.compute_log_prior(coords, inds=inds)
        log_like = self.sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)
        return state

    def run_sampler(
        self,
        nsteps: int,
        initial_state: Optional[State] = None,
        initial_nleaves: Optional[int] = None,
        initial_point: Optional[np.ndarray] = None,
        burn: int = 0,
        thin_by: int = 1,
        progress: bool = True,
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
        initial_nleaves : int, optional
            Number of sources to start with (used if initial_state is None).
        initial_point : array, optional
            Initial parameter values (used if initial_state is None).
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

        if initial_state is None:
            initial_state = self.initialize_state(
                initial_nleaves=initial_nleaves,
                initial_point=initial_point,
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

    def return_sampled_samples(
        self, branch: Optional[str] = None, temperature: int = 0
    ) -> Dict[str, Any]:
        """
        Return the sampled parameter chains.

        Parameters
        ----------
        branch : str, optional
            Branch name to extract. Defaults to first branch.
        temperature : int, optional
            Temperature index to extract. Default 0 (coldest).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'names': List of base parameter names
            - 'labels': LaTeX labels
            - 'chain': Array of shape (nsteps, nwalkers, nleaves_max, ndim) or
              flattened to (n_samples, ndim) excluding NaN entries
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        if branch is None:
            branch = self.branch_names[0]

        chain = self.sampler.get_chain()[branch]
        # Shape: (nsteps, ntemps, nwalkers, nleaves_max, ndim)
        
        # Extract specified temperature
        chain_temp = chain[:, temperature]  # (nsteps, nwalkers, nleaves_max, ndim)

        return {
            "names": self.base_param_names,
            "labels": self.latex_list,
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
            Branch name. Defaults to first branch.
        temperature : int, optional
            Temperature index. Default 0.

        Returns
        -------
        ndarray
            Flattened samples of shape (n_valid_samples, ndim).
        """
        samples = self.return_sampled_samples(branch=branch, temperature=temperature)
        chain = samples["chain"]
        
        # Flatten and remove NaN entries
        flat_chain = chain.reshape(-1, self.ndim)
        valid_mask = ~np.isnan(flat_chain[:, 0])
        
        return flat_chain[valid_mask]

    def return_nleaves(self, branch: Optional[str] = None) -> np.ndarray:
        """
        Return the number of active leaves at each step.

        Parameters
        ----------
        branch : str, optional
            Branch name. Defaults to first branch.

        Returns
        -------
        ndarray
            Array of shape (nsteps, ntemps, nwalkers) with leaf counts.
        """
        if self.sampler is None:
            raise ValueError("Sampler not created. Call create_sampler() first.")

        if branch is None:
            branch = self.branch_names[0]

        return self.sampler.get_nleaves()[branch]

    def return_logZ(self, *, results=None) -> Dict[str, float]:
        """
        Return the log evidence estimate.
        
        Note: Eryn is an MCMC sampler and does not compute Bayesian evidence.
        This method is provided for API consistency but raises NotImplementedError.
        
        Raises
        ------
        NotImplementedError
            Always raised - MCMC samplers do not compute evidence.
        """
        raise NotImplementedError(
            "Eryn RJMCMC is an MCMC sampler and does not compute Bayesian evidence (logZ). "
            "Use nested sampling (Nessai, JAX-NS) for evidence estimates."
        )

    def plot_nleaves_histogram(
        self, figsize: Tuple[int, int] = (6, 12)
    ) -> "plt.Figure":
        """
        Plot histogram of the number of active leaves at each temperature.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default (6, 12).

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt

        nleaves = self.return_nleaves()
        # Shape: (nsteps, ntemps, nwalkers)

        bins = np.arange(self.nleaves_min - 0.5, self.nleaves_max + 1.5)

        fig, axes = plt.subplots(self.ntemps, 1, sharex=True, figsize=figsize)
        if self.ntemps == 1:
            axes = [axes]

        axes[-1].set_xlabel("Number of Models")
        axes[-1].set_xticks(np.arange(self.nleaves_min, self.nleaves_max + 1))

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
            Branch name. Defaults to first branch.
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

        flat_samples = self.return_flat_samples(branch=branch, temperature=temperature)

        fig = corner.corner(
            flat_samples,
            labels=self.latex_list,
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
        print(f"Branch names: {self.branch_names}")
        print(f"Parameters per source: {self.ndim}")
        print(f"Min sources: {self.nleaves_min}")
        print(f"Max sources: {self.nleaves_max}")
        print()
        print("Available configurations:")
        for config in self.rj_model.get_all_configurations():
            likelihood = self.rj_model.get_likelihood_for_config(config)
            n_params = len(likelihood.logL.params)
            print(f"  {config}: {n_params} total parameters")


# Backwards compatibility
ErynRJBridge = DiscoveryErynRJBridge

__all__ = ["RJ_Discovery_model", "DiscoveryErynRJBridge", "ErynRJBridge"]
