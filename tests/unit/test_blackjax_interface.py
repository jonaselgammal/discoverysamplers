#!/usr/bin/env python
"""
Unit tests for BlackJAX nested sampling interface.

Tests the DiscoveryBlackJAXBridge for correct API behavior,
sample structure, and evidence computation.
"""

import pytest
import numpy as np


# ==============================================================================
# Test fixtures
# ==============================================================================

class SimpleLikelihood:
    """Simple 2D Gaussian likelihood for testing."""
    
    def __init__(self):
        self.params = ['x', 'y']
    
    def __call__(self, params):
        x = params['x']
        y = params['y']
        return -0.5 * (x**2 + y**2)


class DiscoveryModel:
    """Wrapper with logL attribute (Discovery-style)."""
    
    def __init__(self):
        self.logL = SimpleLikelihood()


@pytest.fixture
def model():
    """Create a simple test model."""
    return DiscoveryModel()


@pytest.fixture
def priors():
    """Define priors in the format expected by the interfaces."""
    return {
        'x': {'dist': 'uniform', 'min': -5.0, 'max': 5.0},
        'y': {'dist': 'uniform', 'min': -5.0, 'max': 5.0},
    }


@pytest.fixture
def priors_with_fixed():
    """Define priors with a fixed parameter."""
    return {
        'x': {'dist': 'uniform', 'min': -5.0, 'max': 5.0},
        'y': {'dist': 'fixed', 'value': 1.0},
    }


# ==============================================================================
# Helper functions
# ==============================================================================

def validate_samples_dict(result, expected_keys=('names', 'labels', 'chain')):
    """Validate that result dict has expected structure."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    for key in expected_keys:
        assert key in result, f"Missing key '{key}' in result"
    
    assert isinstance(result['names'], list), "names should be a list"
    assert isinstance(result['labels'], list), "labels should be a list"
    assert isinstance(result['chain'], np.ndarray), "chain should be numpy array"
    
    # Chain should be 2D (nsamples, nparams)
    assert result['chain'].ndim == 2, "chain should have 2 dimensions"


def validate_logZ_dict(result):
    """Validate that logZ result dict has expected structure."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'logZ' in result, "Missing 'logZ' key"
    assert 'logZ_err' in result, "Missing 'logZ_err' key"
    assert isinstance(result['logZ'], (int, float)), "logZ should be numeric"
    assert isinstance(result['logZ_err'], (int, float, type(None))), "logZ_err should be numeric or None"


# ==============================================================================
# BlackJAX Bridge Tests
# ==============================================================================

@pytest.mark.skipif(
    not pytest.importorskip("blackjax", reason="blackjax not installed"),
    reason="BlackJAX not available"
)
class TestBlackJAXInterface:
    """Tests for the BlackJAX nested sampling interface."""
    
    @pytest.fixture
    def blackjax_bridge(self, model, priors):
        """Create a BlackJAX bridge and run sampler."""
        from discoverysamplers.blackjax_interface import DiscoveryBlackJAXBridge
        bridge = DiscoveryBlackJAXBridge(model, priors)
        bridge.run_sampler(
            n_live=50,
            num_delete=5,
            max_iterations=100,
            seed=42,
            progress=False
        )
        return bridge
    
    @pytest.fixture
    def blackjax_bridge_fixed(self, model, priors_with_fixed):
        """Create a BlackJAX bridge with fixed parameter."""
        from discoverysamplers.blackjax_interface import DiscoveryBlackJAXBridge
        bridge = DiscoveryBlackJAXBridge(model, priors_with_fixed)
        bridge.run_sampler(
            n_live=50,
            num_delete=5,
            max_iterations=100,
            seed=42,
            progress=False
        )
        return bridge
    
    def test_bridge_creation(self, model, priors):
        """Test that the bridge can be created."""
        from discoverysamplers.blackjax_interface import DiscoveryBlackJAXBridge
        bridge = DiscoveryBlackJAXBridge(model, priors)
        assert bridge.ndim == 2
        assert bridge.sampled_names == ['x', 'y']
    
    def test_return_all_samples(self, blackjax_bridge):
        """Test that return_all_samples returns correct structure."""
        result = blackjax_bridge.return_all_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
        assert 'x' in result['names']
        assert 'y' in result['names']
    
    def test_return_sampled_samples(self, blackjax_bridge):
        """Test that return_sampled_samples returns correct structure."""
        result = blackjax_bridge.return_sampled_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_logZ(self, blackjax_bridge):
        """Test that return_logZ returns correct structure."""
        result = blackjax_bridge.return_logZ()
        validate_logZ_dict(result)
    
    def test_fixed_parameter_handling(self, blackjax_bridge_fixed):
        """Test that fixed parameters are handled correctly."""
        result_sampled = blackjax_bridge_fixed.return_sampled_samples()
        assert len(result_sampled['names']) == 1  # only x is sampled
        assert 'x' in result_sampled['names']
        
        result_all = blackjax_bridge_fixed.return_all_samples()
        assert len(result_all['names']) == 2  # x and y
        
        # Check that y column is all 1.0 (fixed value)
        y_idx = result_all['names'].index('y')
        assert np.allclose(result_all['chain'][:, y_idx], 1.0)
    
    def test_samples_within_bounds(self, blackjax_bridge):
        """Test that samples are within prior bounds."""
        result = blackjax_bridge.return_sampled_samples()
        samples = result['chain']
        
        # All samples should be within [-5, 5]
        assert np.all(samples >= -5.0), "Samples below lower bound"
        assert np.all(samples <= 5.0), "Samples above upper bound"
    
    def test_run_without_results_raises(self, model, priors):
        """Test that accessing results before running raises error."""
        from discoverysamplers.blackjax_interface import DiscoveryBlackJAXBridge
        bridge = DiscoveryBlackJAXBridge(model, priors)
        
        with pytest.raises(RuntimeError, match="No results available"):
            bridge.return_sampled_samples()
        
        with pytest.raises(RuntimeError, match="No results available"):
            bridge.return_all_samples()
        
        with pytest.raises(RuntimeError, match="No results available"):
            bridge.return_logZ()


# ==============================================================================
# Prior transform tests
# ==============================================================================

@pytest.mark.skipif(
    not pytest.importorskip("blackjax", reason="blackjax not installed"),
    reason="BlackJAX not available"
)
class TestPriorTransforms:
    """Tests for prior transform functions."""
    
    def test_uniform_prior_transform(self):
        """Test uniform prior transform."""
        from discoverysamplers.blackjax_interface import _make_prior_transform_blackjax
        from discoverysamplers.priors import _parse_single_prior
        import jax.numpy as jnp
        
        parsed = {'x': _parse_single_prior('x', ('uniform', 0.0, 10.0))}
        transform = _make_prior_transform_blackjax(['x'], parsed)
        
        # Test endpoints
        result_0 = transform(jnp.array([0.0]))
        result_1 = transform(jnp.array([1.0]))
        result_mid = transform(jnp.array([0.5]))
        
        assert float(result_0[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(result_1[0]) == pytest.approx(10.0, abs=1e-6)
        assert float(result_mid[0]) == pytest.approx(5.0, abs=1e-6)
    
    def test_loguniform_prior_transform(self):
        """Test log-uniform prior transform."""
        from discoverysamplers.blackjax_interface import _make_prior_transform_blackjax
        from discoverysamplers.priors import _parse_single_prior
        import jax.numpy as jnp
        
        parsed = {'x': _parse_single_prior('x', ('loguniform', 1.0, 100.0))}
        transform = _make_prior_transform_blackjax(['x'], parsed)
        
        # Test endpoints
        result_0 = transform(jnp.array([0.0]))
        result_1 = transform(jnp.array([1.0]))
        result_mid = transform(jnp.array([0.5]))
        
        assert float(result_0[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(result_1[0]) == pytest.approx(100.0, abs=1e-4)  # Relaxed tolerance for float32
        assert float(result_mid[0]) == pytest.approx(10.0, abs=1e-5)  # geometric mean


# ==============================================================================
# Log prior tests
# ==============================================================================

@pytest.mark.skipif(
    not pytest.importorskip("blackjax", reason="blackjax not installed"),
    reason="BlackJAX not available"
)
class TestLogPrior:
    """Tests for log prior functions."""
    
    def test_uniform_log_prior(self):
        """Test uniform log prior."""
        from discoverysamplers.blackjax_interface import _make_log_prior_blackjax
        from discoverysamplers.priors import _parse_single_prior
        import jax.numpy as jnp
        
        parsed = {'x': _parse_single_prior('x', ('uniform', 0.0, 10.0))}
        log_prior = _make_log_prior_blackjax(['x'], parsed)
        
        # Inside bounds
        lp = log_prior(jnp.array([5.0]))
        assert float(lp) == pytest.approx(-np.log(10.0), abs=1e-6)
        
        # Outside bounds
        lp_out = log_prior(jnp.array([-1.0]))
        assert float(lp_out) == -np.inf
    
    def test_loguniform_log_prior(self):
        """Test log-uniform log prior."""
        from discoverysamplers.blackjax_interface import _make_log_prior_blackjax
        from discoverysamplers.priors import _parse_single_prior
        import jax.numpy as jnp
        
        parsed = {'x': _parse_single_prior('x', ('loguniform', 1.0, 100.0))}
        log_prior = _make_log_prior_blackjax(['x'], parsed)
        
        # Inside bounds
        lp = log_prior(jnp.array([10.0]))
        # log-uniform: p(x) = 1/(x * log(b/a)) => log(p) = -log(x) - log(log(b/a))
        expected = -np.log(10.0) - np.log(np.log(100.0))
        assert float(lp) == pytest.approx(expected, abs=1e-6)
        
        # Outside bounds
        lp_out = log_prior(jnp.array([0.5]))
        assert float(lp_out) == -np.inf


# ==============================================================================
# Integration with other interfaces
# ==============================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("blackjax", reason="blackjax not installed"),
    reason="BlackJAX not available"
)
class TestCrossInterfaceConsistency:
    """Tests that verify BlackJAX results are consistent with other samplers."""
    
    def test_return_keys_match_nessai(self, model, priors, tmp_path):
        """Test that BlackJAX returns same keys as Nessai."""
        from discoverysamplers.blackjax_interface import DiscoveryBlackJAXBridge
        
        try:
            from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
        except ImportError:
            pytest.skip("nessai not installed")
        
        # BlackJAX
        blackjax = DiscoveryBlackJAXBridge(model, priors)
        blackjax.run_sampler(n_live=50, max_iterations=100, seed=42, progress=False)
        blackjax_samples = blackjax.return_all_samples()
        blackjax_logZ = blackjax.return_logZ()
        
        # Nessai
        nessai = DiscoveryNessaiBridge(model, priors)
        nessai.run_sampler(nlive=50, max_iteration=300, output=str(tmp_path))
        nessai_samples = nessai.return_all_samples()
        nessai_logZ = nessai.return_logZ()
        
        # Compare keys
        assert set(blackjax_samples.keys()) == set(nessai_samples.keys())
        assert set(blackjax_logZ.keys()) == set(nessai_logZ.keys())


# ==============================================================================
# Run tests if executed directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
