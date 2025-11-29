"""
Plotting utilities for discoverysamplers.

This module provides common plotting functions that are used across all sampler interfaces.
Each interface provides lightweight wrappers around these functions.

Available Plots
---------------
- **Trace plots**: Show parameter evolution over samples/steps
- **Corner plots**: Show marginal distributions and correlations
- **Run plots**: Diagnostic plots for nested sampling runs

All functions accept a standardized samples dictionary with keys:
- 'names': list of parameter names
- 'labels': list of LaTeX labels for plotting
- 'chain': numpy array of samples

For MCMC chains (Eryn), the chain shape is typically (nsteps, ntemps, nwalkers, nleaves, ndim).
For nested sampling (Nessai, JAX-NS), the chain is (nsamples, ndim).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Type alias for samples dictionary
SamplesDict = Dict[str, Any]


def plot_trace(
    samples: SamplesDict,
    *,
    burn: int = 0,
    fixed_params: Optional[Dict[str, float]] = None,
    fixed_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    alpha: float = 0.3,
    lw: float = 0.7,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Create trace plots showing parameter evolution over samples.
    
    Parameters
    ----------
    samples : dict
        Dictionary with keys 'names', 'labels', 'chain'.
        Chain can be:
        - (nsteps, ntemps, nwalkers, [nleaves,] ndim) for MCMC
        - (nsamples, ndim) for nested sampling
    burn : int, optional
        Number of initial samples to discard, by default 0.
    fixed_params : dict, optional
        Dictionary of fixed parameter values to show as horizontal lines.
    fixed_names : list, optional
        Names of fixed parameters (to identify which to mark).
    figsize : tuple, optional
        Figure size (width, height). Auto-scaled if None.
    alpha : float, optional
        Transparency for trace lines, by default 0.3.
    lw : float, optional
        Line width, by default 0.7.
    colors : list, optional
        Colors for different temperatures/chains. Uses matplotlib defaults if None.
    title : str, optional
        Figure title.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the trace plots.
        
    Examples
    --------
    >>> samples = bridge.return_sampled_samples()
    >>> fig = plot_trace(samples, burn=1000)
    >>> fig.savefig('trace.pdf')
    """
    import matplotlib.pyplot as plt
    
    chain = samples["chain"]
    names = samples["names"]
    labels = samples["labels"]
    
    if fixed_params is None:
        fixed_params = {}
    if fixed_names is None:
        fixed_names = []
    
    # Determine chain shape and handle accordingly
    chain = np.asarray(chain)
    ndim_chain = chain.ndim
    
    # Apply burn-in
    if burn > 0:
        chain = chain[burn:]
    
    n_params = len(names)
    
    # Auto-scale figure size
    if figsize is None:
        figsize = (10, max(2.2, 1.8 * n_params))
    
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    axes_arr = np.atleast_1d(axes)
    
    # Detect chain structure
    # MCMC: (nsteps, ntemps, nwalkers, ndim) or (nsteps, ntemps, nwalkers, nleaves, ndim)
    # Nested: (nsamples, ndim)
    if ndim_chain == 2:
        # Nested sampling: simple 2D chain
        nsamples = chain.shape[0]
        for i, name in enumerate(names):
            ax = axes_arr[i]
            ax.plot(chain[:, i], lw=lw, alpha=0.9, color='C0')
            ax.set_ylabel(labels[i])
            if name in fixed_names and name in fixed_params:
                ax.axhline(fixed_params[name], ls="--", lw=1.0, color="r", label="fixed")
                ax.legend(loc="best", frameon=False)
    
    elif ndim_chain >= 4:
        # MCMC chain with temperatures/walkers
        nsteps = chain.shape[0]
        ntemps = chain.shape[1]
        nwalkers = chain.shape[2]
        
        # Handle nleaves dimension if present
        if ndim_chain == 5:
            # (nsteps, ntemps, nwalkers, nleaves, ndim) - flatten nleaves
            chain = chain.reshape(nsteps, ntemps, nwalkers, -1)
        
        # Set colors
        if colors is None:
            colors = [f'C{j}' for j in range(ntemps)]
        
        for i, name in enumerate(names):
            ax = axes_arr[i]
            ax.set_ylabel(labels[i])
            
            if name in fixed_names and name in fixed_params:
                ax.axhline(fixed_params[name], color='r', linestyle='--', label='Fixed')
                ax.legend(loc="best", frameon=False)
                continue
            
            for j in range(ntemps):
                for k in range(nwalkers):
                    ax.plot(chain[:, j, k, i], alpha=alpha, lw=lw, color=colors[j])
        
        # Add temperature legend if multiple temperatures
        if ntemps > 1:
            handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Temp {i}') 
                      for i in range(ntemps)]
            fig.legend(handles=handles, loc='upper center', ncol=min(ntemps, 8), 
                      bbox_to_anchor=(0.5, 1.02))
    
    else:
        # Fallback: try to plot as-is
        for i, name in enumerate(names):
            ax = axes_arr[i]
            ax.plot(chain[..., i].flatten(), lw=lw, alpha=0.9)
            ax.set_ylabel(labels[i])
    
    axes_arr[-1].set_xlabel('Sample index')
    
    if title:
        fig.suptitle(title, y=1.02)
    
    fig.tight_layout(rect=[0, 0, 1, 0.97] if ndim_chain >= 4 and chain.shape[1] > 1 else None)
    
    return fig


def plot_corner(
    samples: SamplesDict,
    *,
    burn: int = 0,
    temp: int = 0,
    truths: Optional[Sequence[float]] = None,
    quantiles: Optional[Sequence[float]] = None,
    show_titles: bool = True,
    title_fmt: str = ".3f",
    **corner_kwargs,
):
    """
    Create a corner plot showing marginal distributions and correlations.
    
    Parameters
    ----------
    samples : dict
        Dictionary with keys 'names', 'labels', 'chain'.
    burn : int, optional
        Number of initial samples to discard, by default 0.
    temp : int, optional
        Temperature index for MCMC chains (0 = cold chain), by default 0.
    truths : sequence, optional
        True parameter values to mark on the plot.
    quantiles : sequence, optional
        Quantiles to show on 1D histograms. Default is [0.16, 0.5, 0.84].
    show_titles : bool, optional
        Show parameter estimates in titles, by default True.
    title_fmt : str, optional
        Format string for title values, by default ".3f".
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().
        
    Returns
    -------
    matplotlib.figure.Figure
        Corner plot figure.
        
    Examples
    --------
    >>> samples = bridge.return_sampled_samples()
    >>> fig = plot_corner(samples, burn=1000, quantiles=[0.16, 0.5, 0.84])
    >>> fig.savefig('corner.pdf')
    """
    import corner
    
    chain = np.asarray(samples["chain"])
    labels = samples["labels"]
    
    # Apply burn-in
    if burn > 0:
        chain = chain[burn:]
    
    # Handle different chain shapes
    ndim_chain = chain.ndim
    
    if ndim_chain == 2:
        # Nested sampling: (nsamples, ndim)
        flat_chain = chain
    elif ndim_chain >= 4:
        # MCMC: (nsteps, ntemps, nwalkers, ndim) or with nleaves
        if ndim_chain == 5:
            # (nsteps, ntemps, nwalkers, nleaves, ndim)
            chain_temp = chain[:, temp, :, :, :]
            flat_chain = chain_temp.reshape(-1, chain_temp.shape[-1])
            # Remove NaN entries (inactive leaves in RJMCMC)
            valid_mask = ~np.isnan(flat_chain[:, 0])
            flat_chain = flat_chain[valid_mask]
        else:
            # (nsteps, ntemps, nwalkers, ndim)
            flat_chain = chain[:, temp, :, :].reshape(-1, chain.shape[-1])
    else:
        # Fallback
        flat_chain = chain.reshape(-1, chain.shape[-1])
    
    # Set default quantiles
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]
    
    # Create corner plot
    fig = corner.corner(
        flat_chain,
        labels=labels,
        truths=truths,
        quantiles=quantiles,
        show_titles=show_titles,
        title_fmt=title_fmt,
        **corner_kwargs,
    )
    
    return fig


def plot_corner_multi_temp(
    samples: SamplesDict,
    *,
    burn: int = 0,
    temps: Optional[Sequence[int]] = None,
    **corner_kwargs,
):
    """
    Create corner plots for multiple temperatures (MCMC only).
    
    Parameters
    ----------
    samples : dict
        Dictionary with keys 'names', 'labels', 'chain'.
        Chain must have shape (nsteps, ntemps, nwalkers, ndim).
    burn : int, optional
        Number of initial samples to discard, by default 0.
    temps : sequence, optional
        Temperature indices to plot. Default plots all temperatures.
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().
        
    Returns
    -------
    list of matplotlib.figure.Figure
        List of corner plot figures, one per temperature.
    """
    chain = np.asarray(samples["chain"])
    
    if chain.ndim < 4:
        raise ValueError("plot_corner_multi_temp requires MCMC chain with temperature dimension")
    
    ntemps = chain.shape[1]
    
    if temps is None:
        temps = list(range(ntemps))
    
    figs = []
    for temp in temps:
        fig = plot_corner(samples, burn=burn, temp=temp, **corner_kwargs)
        fig.suptitle(f'Temperature {temp}', y=1.02)
        figs.append(fig)
    
    return figs


def plot_run_plot(
    samples: SamplesDict,
    *,
    log_evidence: Optional[float] = None,
    log_evidence_err: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Create a run plot showing sampling progress (nested sampling diagnostics).
    
    This plot shows the log-likelihood values vs sample index, which is useful
    for diagnosing nested sampling convergence.
    
    Parameters
    ----------
    samples : dict
        Dictionary with keys 'names', 'labels', 'chain', and optionally 'log_L'.
    log_evidence : float, optional
        Log evidence estimate to display.
    log_evidence_err : float, optional
        Uncertainty on log evidence.
    figsize : tuple, optional
        Figure size, by default (10, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Run plot figure.
    """
    import matplotlib.pyplot as plt
    
    chain = np.asarray(samples["chain"])
    nsamples = chain.shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot sample indices
    ax.plot(range(nsamples), 'b-', lw=0.5, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sample index')
    
    # Add evidence annotation if provided
    if log_evidence is not None:
        text = f'log Z = {log_evidence:.2f}'
        if log_evidence_err is not None:
            text += f' ± {log_evidence_err:.2f}'
        ax.text(0.95, 0.95, text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    return fig


def plot_nleaves_histogram(
    nleaves: np.ndarray,
    *,
    nleaves_min: int,
    nleaves_max: int,
    true_nleaves: Optional[int] = None,
    temp: int = 0,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """
    Plot histogram of number of active components (for RJMCMC).
    
    Parameters
    ----------
    nleaves : ndarray
        Array of nleaves values with shape (nsteps, ntemps, nwalkers) or (nsteps, ntemps).
    nleaves_min : int
        Minimum number of leaves.
    nleaves_max : int
        Maximum number of leaves.
    true_nleaves : int, optional
        True number of components to mark on plot.
    temp : int, optional
        Temperature index to use (0 = cold chain), by default 0.
    figsize : tuple, optional
        Figure size, by default (8, 5).
    title : str, optional
        Plot title.
        
    Returns
    -------
    matplotlib.figure.Figure
        Histogram figure.
        
    Examples
    --------
    >>> nleaves = bridge.return_nleaves()
    >>> fig = plot_nleaves_histogram(nleaves, nleaves_min=1, nleaves_max=5, true_nleaves=2)
    """
    import matplotlib.pyplot as plt
    
    nleaves = np.asarray(nleaves)
    
    # Extract cold chain
    if nleaves.ndim >= 2:
        nleaves_flat = nleaves[:, temp].flatten()
    else:
        nleaves_flat = nleaves.flatten()
    
    bins = np.arange(nleaves_min - 0.5, nleaves_max + 1.5)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.hist(nleaves_flat, bins=bins, edgecolor='black', alpha=0.7)
    
    if true_nleaves is not None:
        ax.axvline(true_nleaves, color='r', linestyle='--', lw=2, 
                  label=f'True ({true_nleaves} sources)')
        ax.legend()
    
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Count')
    ax.set_xticks(range(nleaves_min, nleaves_max + 1))
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Model dimension posterior')
    
    fig.tight_layout()
    return fig


def plot_parameter_summary(
    samples: SamplesDict,
    *,
    burn: int = 0,
    credible_interval: float = 0.9,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Create a summary plot showing parameter estimates with credible intervals.
    
    Parameters
    ----------
    samples : dict
        Dictionary with keys 'names', 'labels', 'chain'.
    burn : int, optional
        Number of initial samples to discard, by default 0.
    credible_interval : float, optional
        Credible interval width (0-1), by default 0.9.
    figsize : tuple, optional
        Figure size. Auto-scaled if None.
        
    Returns
    -------
    matplotlib.figure.Figure
        Summary plot figure.
    """
    import matplotlib.pyplot as plt
    
    chain = np.asarray(samples["chain"])
    names = samples["names"]
    labels = samples["labels"]
    
    # Apply burn-in
    if burn > 0:
        chain = chain[burn:]
    
    # Flatten chain if needed
    if chain.ndim > 2:
        chain = chain.reshape(-1, chain.shape[-1])
    
    n_params = len(names)
    
    # Calculate quantiles
    alpha = (1 - credible_interval) / 2
    lower = np.percentile(chain, alpha * 100, axis=0)
    median = np.percentile(chain, 50, axis=0)
    upper = np.percentile(chain, (1 - alpha) * 100, axis=0)
    
    if figsize is None:
        figsize = (8, max(3, 0.5 * n_params))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    y_positions = np.arange(n_params)
    
    # Plot error bars
    ax.errorbar(
        median, y_positions,
        xerr=[median - lower, upper - median],
        fmt='o', capsize=4, capthick=1.5, markersize=6,
        color='C0', ecolor='C0'
    )
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Parameter value')
    ax.set_title(f'Parameter estimates ({credible_interval*100:.0f}% CI)')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    return fig


__all__ = [
    "plot_trace",
    "plot_corner", 
    "plot_corner_multi_temp",
    "plot_run_plot",
    "plot_nleaves_histogram",
    "plot_parameter_summary",
]
