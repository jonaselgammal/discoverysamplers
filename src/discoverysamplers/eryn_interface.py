# Filename: eryn.py
# Author: Jonas El Gammal
# Description: Interface to Eryn
# Date Created: 2025-09-01

"""
Bridge utilities to run Eryn MCMC on Discovery likelihoods (PTAs).

Tested with:
  - discovery 0.5 (JAX-based PTA analysis)  
  - eryn >= 1.2 (emcee-like API)

This file provides:
  - DiscoveryErynBridge: packs/unpacks parameter dicts, computes log-prob

Notes:
  • Discovery’s likelihoods are JAX-ready callables that accept a dict of
    named parameters. We wrap them with a flat θ-vector interface that Eryn
    expects.
  • Priors: by default, uniform within bounds; you can supply custom
    log-prior callables per parameter.
  • Initialization: walkers are drawn from priors (or a Gaussian ball around
    an initial point if provided).
  • Sampling: uses Eryn’s EnsembleSampler with optional parallel tempering.

Gotchas:
    • Make sure Discovery and Eryn are installed in your Python environment.
    • Ensure parameter names in priors match those in the Discovery model.
    • If no parameters are sampled (all fixed), Eryn will raise an error.
    • Currently, the interface assumes that there is only one model, i.e. no
      reversible-jump sampling.
"""


import numpy as np
import re
from typing import Mapping, Sequence, Optional, Dict, List, Any
import warnings
import sys ### REMOVE ME

try:
    from eryn.prior import uniform_dist, log_uniform, ProbDistContainer
    from eryn.ensemble import EnsembleSampler
except ImportError:
    raise ImportError("eryn is not installed. Please install it to use this module.")


class DiscoveryErynBridge:
    def __init__(self, model, priors: Optional[Any] = None, latex_labels: Optional[Dict[str, str]] = None):
        """
        Initialize the Eryn interface for Discovery models.

        This class creates an interface between Discovery models and the Eryn sampler,
        handling parameter management, prior specifications, and likelihood calculations.

        Parameters
        ----------
        model : object
            Discovery model object that must implement:
            - model.logL(params: dict) -> float : likelihood function
            - model.params : model parameters

        priors : None | list[Param] | dict, optional
            Prior specifications for model parameters. Can be:
            - None: Uses default priors from the model if available
            - list[Param]: List of parameter specifications (legacy format)
            - dict: Cobaya-style prior specifications {name: {dist:..., ...}}

        latex_labels : dict, optional
            Dictionary mapping parameter names to their LaTeX representations
            for plotting and display purposes. If not provided, parameter
            names are used as labels.

        Attributes
        ----------
        discovery_paramnames : list
            List of all parameter names in model order
        sampled_prior_dict : dict
            Dictionary of prior specifications for sampled parameters
        fixed_param_dict : dict
            Dictionary of fixed parameter values
        fixed_names : list
            Names of fixed parameters
        sampled_names : list
            Names of sampled parameters
        n_fixed : int
            Number of fixed parameters
        n_sampled : int
            Number of sampled parameters
        ndim : int
            Dimension of parameter space (same as n_sampled)
        eryn_mapping : dict
            Maps parameter names to their indices in the θ-vector
        eryn_prior_dict : dict
            Prior specifications mapped to θ-vector indices
        eryn_prior_container : ProbDistContainer
            Eryn prior object for sampled parameters
        latex_labels : dict
            Mapping of parameter names to LaTeX labels
        latex_list : list
            LaTeX labels for all parameters
        sampled_names_latex : list
            LaTeX labels for sampled parameters
        fixed_names_latex : list
            LaTeX labels for fixed parameters

        Raises
        ------
        ValueError
            If any model parameters are missing from the prior specifications
        """
        self.model = model

        # 1) Get all model parameter names (order = sampler order)
        self.discovery_paramnames = self._infer_model_param_names(model)

        # 2) Normalize priors to a single internal spec per parameter
        self.sampled_prior_dict, self.fixed_param_dict = self._check_and_create_priors(priors, model, self.discovery_paramnames)

        # 3) Sampled vs fixed parameters
        self.fixed_names = list(self.fixed_param_dict.keys())
        self.sampled_names = list(self.sampled_prior_dict.keys())

        # make sure that the sampled + fixed = all parameters
        all_names = set(self.fixed_names).union(set(self.sampled_names))
        if set(self.discovery_paramnames) != all_names:
            missing = set(self.discovery_paramnames) - all_names
            raise ValueError(f"Parameters missing from priors: {missing}. Please provide priors for all parameters.")
        # Make sure the order is the same as in discovery_paramnames
        self.fixed_names.sort(key=self.discovery_paramnames.index)
        self.sampled_names.sort(key=self.discovery_paramnames.index)

        self.n_fixed = len(self.fixed_names)
        self.n_sampled = len(self.sampled_names)
        self.ndim = self.n_sampled  # for Eryn

        # map from sampled param name to index in θ-vector
        self.eryn_mapping = {name: i for i, name in enumerate(self.sampled_names)}

        # Create a eryn prior dict that maps the names to the θ-vector
        self.eryn_prior_dict = {self.eryn_mapping[name]: self.sampled_prior_dict[name] for name in self.sampled_names}

        # 4) Build Eryn prior object for sampled parameters
        self.eryn_prior_container = ProbDistContainer(self.eryn_prior_dict)

        if latex_labels is None:
            latex_labels = {}

        self.latex_labels = {name: latex_labels.get(name, name) for name in self.discovery_paramnames}

        self.latex_list = [self.latex_labels[name] for name in self.discovery_paramnames]
        self.sampled_names_latex = [self.latex_labels[name] for name in self.sampled_names]
        self.fixed_names_latex = [self.latex_labels[name] for name in self.fixed_names]

        # 5) Likelihood function (Discovery accepts a dict)
        # JIT optional — your earlier code used jax, keep the same idea if you want.
        self._loglike_fn = model.logL  # must accept a dict of {name: value}

    def create_sampler(self, nwalkers, **kwargs):
        """Create an ensemble sampler for MCMC sampling.
        This method initializes an EnsembleSampler object for Markov Chain Monte Carlo sampling,
        using the provided likelihood function and priors.
        Parameters
        ----------
        nwalkers : int
            Number of walkers to use in the ensemble sampler
        **kwargs : dict
            Additional keyword arguments to pass to the EnsembleSampler constructor
        Returns
        -------
        EnsembleSampler
            Initialized ensemble sampler object
        Raises
        ------
        ValueError
            If no parameters are marked for sampling (ndim = 0)
        Notes
        -----
        The method creates an internal likelihood function that combines both fixed and
        sampled parameters before evaluation. The sampler is stored as an instance
        attribute and the initial shape for p0 is recorded.
        """

        if self.ndim == 0:
            raise ValueError("No sampled parameters. Provide priors that mark at least one parameter as non-fixed.")
        
        def _loglike_only(theta):
            # Merge fixed + current sampled theta
            params = self.unpack(theta)  # sampled
            params.update(self.fixed_param_dict)  # add fixed
            val = float(self._loglike_fn(params))
            return val

        self.sampler = EnsembleSampler(
            nwalkers,
            self.ndim,
            _loglike_only,
            priors=self.eryn_prior_container,
            **kwargs
        )

        self.p0_shape = self.sampler.backend.shape["model_0"][:-1]

        return self.sampler
    
    def run_sampler(self, nsteps, p0=None, **kwargs):
        """
        Run the MCMC sampler for a specified number of steps.
        This method executes the MCMC sampling process using the previously created sampler.
        It can start from provided initial positions or generate them from the prior distributions.
        Parameters
        ----------
        nsteps : int
            Number of steps to run the MCMC sampler
        p0 : array-like, optional
            Initial positions for the walkers. If None, positions are drawn from the prior
            distributions. Shape should match sampler requirements.
        **kwargs : dict
            Additional keyword arguments to pass to the sampler's run_mcmc method
        Returns
        -------
        sampler : object
            The MCMC sampler object after running the chain
        Raises
        ------
        ValueError
            If the sampler has not been created or if no parameters are marked for sampling
        Notes
        -----
        The method requires that create_sampler has been called first and that at least
        one parameter has been marked as non-fixed in the prior distributions.
        """

        if self.sampler is None:
            raise ValueError("Sampler not created. Call the create_sampler method first.")

        if self.ndim == 0:
            raise ValueError("No sampled parameters. Provide priors that mark at least one parameter as non-fixed.")

        # Initial positions from priors using Eryn's prior container
        if p0 is None:
            p0 = self.eryn_prior_container.rvs(size=self.p0_shape)

        self.sampler.run_mcmc(p0, nsteps, **kwargs)
        return self.sampler
    
    def return_all_samples(self):
        """
        Returns all MCMC samples including both sampled and fixed parameters.
        This method retrieves the MCMC chain from the sampler and combines the sampled parameters
        with the fixed parameters to create a complete parameter set for each sample.
        Returns
        -------
        dict
            A dictionary containing:
            - 'names' (list): Names of all parameters (sampled and fixed)
            - 'labels' (list): LaTeX labels for all parameters
            - 'chain' (ndarray): Array of shape (nwalkers*nsteps, n_all_params) containing
              all parameter samples, where n_all_params is the total number of parameters
              (both sampled and fixed)
        Raises
        ------
        ValueError
            If the sampler has not been created using create_sampler()
        RuntimeError
            If the MCMC chain cannot be retrieved (e.g., if sampling hasn't been run)
        """

        if self.sampler is None:
            raise ValueError("Sampler not created. Call the create_sampler method first.")
        try:
            chain = self.sampler.get_chain()["model_0"]  # shape (nwalkers*nsteps, ndim)
        except Exception as e:
            raise RuntimeError("Could not get chain from sampler. Make sure sampling has been run.") from e
        chain_shape = chain.shape

        all_samples = np.zeros(chain_shape[:-1] + (len(self.discovery_paramnames),), dtype=float)

        # Fill in sampled parameters
        for i, name in enumerate(self.sampled_names):
            idx = self.discovery_paramnames.index(name)
            all_samples[..., idx] = chain[..., i]

        # Fill in fixed parameters
        for name in self.fixed_names:
            idx = self.discovery_paramnames.index(name)
            all_samples[..., idx] = self.fixed_param_dict[name]

        return {"names": self.discovery_paramnames, "labels": self.latex_list, "chain": all_samples}  # shape (nwalkers*nsteps, n_all_params)

    def return_sampled_samples(self):
        """
        Returns the sampled parameters and their names from the MCMC chain.
        This method retrieves the sampling chain from the sampler and returns it along with 
        parameter names and their LaTeX representations.
        Returns
        -------
        dict
            Dictionary containing:
            - 'names' (list): List of parameter names
            - 'labels' (list): List of parameter names in LaTeX format
            - 'chain' (ndarray): MCMC chain with shape (nwalkers*nsteps, n_sampled_params)
        Raises
        ------
        ValueError
            If sampler has not been created using create_sampler() method
        RuntimeError
            If sampling has not been run or chain cannot be retrieved
        Notes
        -----
        The returned chain combines all walkers and steps into a single array, flattening
        the typical (nwalkers, nsteps, ndim) shape into (nwalkers*nsteps, ndim).
        """

        if self.sampler is None:
            raise ValueError("Sampler not created. Call the create_sampler method first.")
        try:
            chain = self.sampler.get_chain()["model_0"]  # shape (nwalkers*nsteps, ndim)
        except Exception as e:
            raise RuntimeError("Could not get chain from sampler. Make sure sampling has been run.") from e
        return {"names": self.sampled_names, "labels": self.sampled_names_latex, "chain": chain}  # shape (nwalkers*nsteps, n_sampled_params)

    def plot_trace(self, burn=0, plot_fixed=False):
        """Plot the MCMC chains for all parameters.
        This method creates a plot showing the evolution of all parameter chains across steps, 
        temperatures and walkers.
        Parameters
        ----------
        burn : int, optional
            Number of initial steps to discard from the plot, by default 0
        plot_fixed : bool, optional
            If True, includes fixed parameters in the plot, by default False
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the trace plots
        Notes
        -----
        - Each parameter is plotted in a separate subplot
        - For fixed parameters, their fixed values are shown as horizontal red dashed lines
        - For multiple temperatures, different colors are used for each temperature chain
        - Multiple walkers are plotted with low opacity to show the overall evolution
        - With multiple temperatures, a legend is added above the figure showing temperature indices
        Examples
        --------
        >>> sampler.plot_trace(burn=1000)  # Plot chains excluding first 1000 steps
        >>> sampler.plot_trace(plot_fixed=True)  # Include fixed parameters in plot
        """

        import matplotlib.pyplot as plt

        if plot_fixed:
            samples = self.return_all_samples()
        else:
            samples = self.return_sampled_samples()
        chain = samples["chain"][burn:] 
        names = samples["names"]
        labels = samples["labels"]

        ntemps = chain.shape[1]
        nwalkers = chain.shape[2]

        n_params = len(names)
        fig, axes = plt.subplots(n_params, 1, figsize=(8, 2 * n_params), sharex=True)

        for i, name in enumerate(names):
            ax = axes[i] if n_params > 1 else axes
            ax.set_ylabel(labels[i])
            if name in self.fixed_names:
                ax.axhline(self.fixed_param_dict[name], color='r', linestyle='--', label='Fixed value')
                ax.legend()
                continue
            for j in range(ntemps):
                for k in range(nwalkers):
                    ax.plot(chain[:, j, k, :, i].reshape(-1), alpha=0.1, color=f'C{j}')
        axes[-1].set_xlabel('Step number')
        # If there is more than 1 temperature, add a legend on top of the figure outside the axes
        if ntemps > 1:
            handles = [plt.Line2D([0], [0], color=f'C{i}', lw=2, label=f'Temp {i}') for i in range(ntemps)]
            fig.legend(handles=handles, loc='upper center', ncol=ntemps, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig
    
    def plot_corner(self, burn=0, **kwargs):
        """
        Create corner plots for the MCMC chain at each temperature.
        This method generates corner plots using the 'corner' package, displaying the 
        marginal distributions and correlations between parameters for each temperature
        in the MCMC sampling.
        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard as burn-in period. Default is 0.
        **kwargs : dict
            Additional keyword arguments to pass to corner.corner().
        Returns
        -------
        list of matplotlib.figure.Figure
            List of corner plot figures, one for each temperature in the chain.
        Notes
        -----
        The corner plots show the marginalized posterior distributions for each parameter
        along the diagonal, and the 2D projections of the posterior probability 
        distributions for each pair of parameters in the off-diagonal panels.
        Requires the 'corner' package to be installed.
        """

        import corner

        samples = self.return_sampled_samples()
        chain = samples["chain"][burn:] 
        labels = samples["labels"]

        ntemps = chain.shape[1]
        nwalkers = chain.shape[2]

        # Make a corner plot for each temperature since they have different distributions
        figs = []
        for i in range(ntemps):

            flat_chain = chain[:, i, :, :].reshape(-1, self.ndim)
            fig = corner.corner(flat_chain, labels=labels, **kwargs)
            figs.append(fig)
        return figs

        return fig

    # ----- Packing helpers (sampled only) ----- #
    def names(self) -> List[str]:
        return list(self.sampled_names)

    def pack(self, d: Mapping[str, float]) -> np.ndarray:
        return np.array([float(d[n]) for n in self.sampled_names], dtype=float)

    def unpack(self, theta: Sequence[float]) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.sampled_names, theta)}
    
    def pack_all(self, d: Mapping[str, float]) -> np.ndarray:
        return np.array([float(d[n]) for n in self.discovery_paramnames], dtype=float)

    def unpack_all(self, theta: Sequence[float]) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.discovery_paramnames, theta)}

    # ====================================================================== #
    # Internals
    # ====================================================================== #

    def _infer_model_param_names(self, model) -> List[str]:
        """Infer parameter names from the model object."""
        names = []
        if hasattr(model.logL, "params"):
            p = getattr(model.logL, "params")
            if isinstance(p, (list, tuple)) and all(isinstance(x, str) for x in p):
                names = list(p)
            else:
                raise ValueError("model.logL.params doesn't seem to exist or return a list of strings. Please provide a correct discovery model.")
        return names

    def _check_and_create_priors(self, priors, model, discovery_paramnames) -> dict:
        """Check and create prior distributions for the model parameters."""
        try:
            from discovery.prior import priordict_standard
        except ImportError:
            warnings.warn("Could not find default priors in Discovery. You must provide priors explicitly.")
            priordict_standard = {}

        if priors is None:
            sampled_prior_dict = {}
            for parname in discovery_paramnames:
                for par, range in priordict_standard.items():
                    if re.match(par, parname):
                        sampled_prior_dict[parname] = uniform_dist(range[0], range[1])
                        break
                else:
                    raise KeyError(f"No known prior for {parname}. Please provide priors explicitly.")
            return sampled_prior_dict, {}

        elif isinstance(priors, dict):
            keys = priors.keys()
            missing = [par for par in discovery_paramnames if par not in keys]
            sampled_prior_dict = {}
            fixed_params = {}
            if missing:
                raise ValueError(f"Priors missing for parameters: {missing}. You can provide None if ",
                                 "you want to use default priors from the model but all parameters must be covered.")
            for i, parname in enumerate(discovery_paramnames):
                spec = priors[parname]
                if spec is None or spec == "default":

                    for par, range in priordict_standard.items():
                        if re.match(par, parname):
                            sampled_prior_dict[parname] = uniform_dist(range[0], range[1])
                            break
                    else:
                        raise KeyError(f"No known default prior for {parname}. Please provide priors explicitly.")
                elif isinstance(spec, object) and not isinstance(spec, dict):
                    if hasattr(spec, 'logpdf') and hasattr(spec, 'rvs'):
                        sampled_prior_dict[parname] = spec
                    else:
                        raise ValueError(f"Prior object for {parname} must have logpdf and rvs methods.")
                elif 'dist' not in spec:
                    raise ValueError(f"Prior for {parname} missing 'dist' key.")
                if spec['dist'] == 'uniform':
                    if 'min' not in spec or 'max' not in spec:
                        raise ValueError(f"Uniform prior for {parname} requires 'min' and 'max'.")
                    sampled_prior_dict[parname] = uniform_dist(spec['min'], spec['max'])
                elif spec['dist'] == 'loguniform':
                    if 'a' not in spec or 'b' not in spec:
                        raise ValueError(f"Log-uniform prior for {parname} requires 'a' and 'b'.")
                    sampled_prior_dict[parname] = log_uniform(spec['a'], spec['b'])
                elif spec['dist'] == 'fixed':
                    if 'value' not in spec:
                        raise ValueError(f"Fixed prior for {parname} requires 'value'.")
                    fixed_params[parname] = spec['value']
                else:
                    raise ValueError(f"Unsupported prior dist '{spec['dist']}' for {parname}. Supported: uniform, log_uniform, fixed.")
        return sampled_prior_dict, fixed_params