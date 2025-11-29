"""
Test plotting functions and generate example plots for documentation.

This module tests all plotting functions in discoverysamplers.plots and
generates PDF figures that can be used in the documentation.

Usage:
    pytest tests/unit/test_plotting.py -v
    
    # Or run directly to generate plots:
    python tests/unit/test_plotting.py
"""
import os
import sys
import numpy as np
import pytest

# Output directory for plots
PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plot_outputs")


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for plots."""
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    return PLOT_OUTPUT_DIR


@pytest.fixture
def mock_nested_samples():
    """Create mock nested sampling results (2D chain)."""
    np.random.seed(42)
    nsamples = 2000
    ndim = 3
    
    # Generate correlated samples (mock posterior)
    mean = [0.5, 1.0, -0.5]
    cov = [[0.1, 0.05, 0.02],
           [0.05, 0.2, -0.03],
           [0.02, -0.03, 0.15]]
    chain = np.random.multivariate_normal(mean, cov, size=nsamples)
    
    return {
        "names": ["param_a", "param_b", "param_c"],
        "labels": [r"$\alpha$", r"$\beta$", r"$\gamma$"],
        "chain": chain,
    }


@pytest.fixture
def mock_mcmc_samples():
    """Create mock MCMC results with temperatures and walkers."""
    np.random.seed(42)
    nsteps = 500
    ntemps = 2
    nwalkers = 16
    ndim = 3
    
    # Generate MCMC-like chain with shape (nsteps, ntemps, nwalkers, ndim)
    chain = np.zeros((nsteps, ntemps, nwalkers, ndim))
    
    # Cold chain (temp 0) - concentrated around true values
    true_vals = [0.5, 1.0, -0.5]
    for i in range(ndim):
        for w in range(nwalkers):
            # Random walk with burn-in
            x = true_vals[i] + 0.5 * np.random.randn()
            for s in range(nsteps):
                x += 0.02 * np.random.randn()
                chain[s, 0, w, i] = x
    
    # Hot chain (temp 1) - more spread out
    for i in range(ndim):
        for w in range(nwalkers):
            x = true_vals[i] + 1.0 * np.random.randn()
            for s in range(nsteps):
                x += 0.1 * np.random.randn()
                chain[s, 1, w, i] = x
    
    return {
        "names": ["param_a", "param_b", "param_c"],
        "labels": [r"$\alpha$", r"$\beta$", r"$\gamma$"],
        "chain": chain,
    }


@pytest.fixture
def mock_rjmcmc_nleaves():
    """Create mock RJMCMC nleaves array."""
    np.random.seed(42)
    nsteps = 1000
    ntemps = 2
    nwalkers = 16
    
    # Simulate nleaves with preference for 2 components
    nleaves = np.zeros((nsteps, ntemps, nwalkers), dtype=int)
    
    # Cold chain - mostly 2 components
    probs = [0.1, 0.6, 0.25, 0.05]  # P(1), P(2), P(3), P(4)
    for s in range(nsteps):
        for w in range(nwalkers):
            nleaves[s, 0, w] = np.random.choice([1, 2, 3, 4], p=probs)
    
    # Hot chain - more uniform
    probs_hot = [0.25, 0.35, 0.25, 0.15]
    for s in range(nsteps):
        for w in range(nwalkers):
            nleaves[s, 1, w] = np.random.choice([1, 2, 3, 4], p=probs_hot)
    
    return nleaves


class TestPlotTrace:
    """Test trace plotting functionality."""
    
    def test_trace_nested_sampling(self, mock_nested_samples, output_dir):
        """Test trace plot with nested sampling (2D) chain."""
        from discoverysamplers.plots import plot_trace
        
        fig = plot_trace(mock_nested_samples)
        
        assert fig is not None
        assert len(fig.axes) == 3  # One axis per parameter
        
        # Save for documentation
        fig.savefig(os.path.join(output_dir, "trace_nested.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "trace_nested.png"), dpi=150, bbox_inches="tight")
        
    def test_trace_mcmc(self, mock_mcmc_samples, output_dir):
        """Test trace plot with MCMC (4D) chain."""
        from discoverysamplers.plots import plot_trace
        
        fig = plot_trace(mock_mcmc_samples, burn=50)
        
        assert fig is not None
        assert len(fig.axes) == 3
        
        # Save for documentation
        fig.savefig(os.path.join(output_dir, "trace_mcmc.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "trace_mcmc.png"), dpi=150, bbox_inches="tight")
    
    def test_trace_with_fixed_params(self, mock_nested_samples, output_dir):
        """Test trace plot showing fixed parameter values."""
        from discoverysamplers.plots import plot_trace
        
        fig = plot_trace(
            mock_nested_samples,
            fixed_params={"param_b": 1.0},
            fixed_names=["param_b"],
        )
        
        assert fig is not None
        fig.savefig(os.path.join(output_dir, "trace_with_fixed.pdf"), bbox_inches="tight")


class TestPlotCorner:
    """Test corner plot functionality."""
    
    def test_corner_nested_sampling(self, mock_nested_samples, output_dir):
        """Test corner plot with nested sampling chain."""
        from discoverysamplers.plots import plot_corner
        
        fig = plot_corner(
            mock_nested_samples,
            truths=[0.5, 1.0, -0.5],
            quantiles=[0.16, 0.5, 0.84],
        )
        
        assert fig is not None
        
        # Save for documentation
        fig.savefig(os.path.join(output_dir, "corner_nested.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "corner_nested.png"), dpi=150, bbox_inches="tight")
    
    def test_corner_mcmc(self, mock_mcmc_samples, output_dir):
        """Test corner plot with MCMC chain (cold chain only)."""
        from discoverysamplers.plots import plot_corner
        
        fig = plot_corner(
            mock_mcmc_samples,
            burn=100,
            temp=0,  # Cold chain
            truths=[0.5, 1.0, -0.5],
        )
        
        assert fig is not None
        fig.savefig(os.path.join(output_dir, "corner_mcmc.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "corner_mcmc.png"), dpi=150, bbox_inches="tight")
    
    def test_corner_multi_temp(self, mock_mcmc_samples, output_dir):
        """Test corner plots for multiple temperatures."""
        from discoverysamplers.plots import plot_corner_multi_temp
        
        figs = plot_corner_multi_temp(mock_mcmc_samples, burn=100, temps=[0, 1])
        
        assert len(figs) == 2
        
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(output_dir, f"corner_temp{i}.pdf"), bbox_inches="tight")


class TestPlotNleaves:
    """Test nleaves histogram plotting for RJMCMC."""
    
    def test_nleaves_histogram(self, mock_rjmcmc_nleaves, output_dir):
        """Test nleaves histogram plot."""
        from discoverysamplers.plots import plot_nleaves_histogram
        
        fig = plot_nleaves_histogram(
            mock_rjmcmc_nleaves,
            nleaves_min=1,
            nleaves_max=4,
            true_nleaves=2,
            temp=0,
        )
        
        assert fig is not None
        
        # Save for documentation
        fig.savefig(os.path.join(output_dir, "nleaves_histogram.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "nleaves_histogram.png"), dpi=150, bbox_inches="tight")


class TestPlotParameterSummary:
    """Test parameter summary plot."""
    
    def test_parameter_summary(self, mock_nested_samples, output_dir):
        """Test parameter summary plot with credible intervals."""
        from discoverysamplers.plots import plot_parameter_summary
        
        fig = plot_parameter_summary(
            mock_nested_samples,
            credible_interval=0.9,
        )
        
        assert fig is not None
        fig.savefig(os.path.join(output_dir, "parameter_summary.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, "parameter_summary.png"), dpi=150, bbox_inches="tight")


def generate_all_plots():
    """Generate all plots for documentation (run as script)."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create output directory
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Generating plots in: {PLOT_OUTPUT_DIR}")
    
    # Generate fixtures
    np.random.seed(42)
    
    # Nested sampling samples
    nsamples = 2000
    mean = [0.5, 1.0, -0.5]
    cov = [[0.1, 0.05, 0.02],
           [0.05, 0.2, -0.03],
           [0.02, -0.03, 0.15]]
    nested_chain = np.random.multivariate_normal(mean, cov, size=nsamples)
    nested_samples = {
        "names": ["param_a", "param_b", "param_c"],
        "labels": [r"$\alpha$", r"$\beta$", r"$\gamma$"],
        "chain": nested_chain,
    }
    
    # MCMC samples
    nsteps, ntemps, nwalkers, ndim = 500, 2, 16, 3
    mcmc_chain = np.zeros((nsteps, ntemps, nwalkers, ndim))
    true_vals = [0.5, 1.0, -0.5]
    for i in range(ndim):
        for w in range(nwalkers):
            x = true_vals[i] + 0.5 * np.random.randn()
            for s in range(nsteps):
                x += 0.02 * np.random.randn()
                mcmc_chain[s, 0, w, i] = x
            x = true_vals[i] + 1.0 * np.random.randn()
            for s in range(nsteps):
                x += 0.1 * np.random.randn()
                mcmc_chain[s, 1, w, i] = x
    mcmc_samples = {
        "names": ["param_a", "param_b", "param_c"],
        "labels": [r"$\alpha$", r"$\beta$", r"$\gamma$"],
        "chain": mcmc_chain,
    }
    
    # nleaves for RJMCMC
    nleaves = np.zeros((1000, 2, 16), dtype=int)
    probs = [0.1, 0.6, 0.25, 0.05]
    for s in range(1000):
        for w in range(16):
            nleaves[s, 0, w] = np.random.choice([1, 2, 3, 4], p=probs)
            nleaves[s, 1, w] = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
    
    # Import plotting functions
    from discoverysamplers.plots import (
        plot_trace, plot_corner, plot_corner_multi_temp,
        plot_nleaves_histogram, plot_parameter_summary
    )
    
    # Generate plots
    print("Generating trace_nested...")
    fig = plot_trace(nested_samples)
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "trace_nested.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "trace_nested.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print("Generating trace_mcmc...")
    fig = plot_trace(mcmc_samples, burn=50)
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "trace_mcmc.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "trace_mcmc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print("Generating corner_nested...")
    fig = plot_corner(nested_samples, truths=[0.5, 1.0, -0.5], quantiles=[0.16, 0.5, 0.84])
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "corner_nested.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "corner_nested.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print("Generating corner_mcmc...")
    fig = plot_corner(mcmc_samples, burn=100, temp=0, truths=[0.5, 1.0, -0.5])
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "corner_mcmc.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "corner_mcmc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print("Generating nleaves_histogram...")
    fig = plot_nleaves_histogram(nleaves, nleaves_min=1, nleaves_max=4, true_nleaves=2)
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "nleaves_histogram.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "nleaves_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print("Generating parameter_summary...")
    fig = plot_parameter_summary(nested_samples, credible_interval=0.9)
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "parameter_summary.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_OUTPUT_DIR, "parameter_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nAll plots saved to: {PLOT_OUTPUT_DIR}")
    print("Files generated:")
    for f in sorted(os.listdir(PLOT_OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    generate_all_plots()
