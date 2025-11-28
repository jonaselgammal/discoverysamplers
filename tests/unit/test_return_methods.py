#!/usr/bin/env python
"""
Unit tests for sampler interface return methods.

Tests that return_all_samples, return_sampled_samples, and return_logZ
work consistently across all sampler interfaces.
"""

import pytest
import numpy as np
import tempfile
import shutil


# ==============================================================================
# Fixtures
# ==============================================================================

class SimpleLikelihood:
    """Simple 2D Gaussian likelihood for testing."""
    
    def __init__(self):
        # Parameter names - required by Eryn interface
        self.params = ['x', 'y']
    
    def __call__(self, params):
        x = params['x']
        y = params['y']
        return -0.5 * (x**2 + y**2)  # Gaussian centered at origin


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
def temp_output_dir():
    """Create a temporary directory for sampler output."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


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
    
    # Chain should be 2D
    assert result['chain'].ndim >= 1, "chain should have at least 1 dimension"


def validate_logZ_dict(result):
    """Validate that logZ result dict has expected structure."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'logZ' in result, "Missing 'logZ' key"
    assert 'logZ_err' in result, "Missing 'logZ_err' key"
    assert isinstance(result['logZ'], (int, float)), "logZ should be numeric"
    assert isinstance(result['logZ_err'], (int, float)), "logZ_err should be numeric"


# ==============================================================================
# JAX-NS Tests
# ==============================================================================

@pytest.mark.slow
class TestJAXNSInterface:
    """Tests for the JAX-NS sampler interface."""
    
    @pytest.fixture
    def jaxns_bridge(self, model, priors):
        """Create a JAX-NS bridge and run sampler."""
        from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
        bridge = DiscoveryJAXNSBridge(model, priors)
        bridge.run_sampler(nlive=50, max_samples=500)
        return bridge
    
    def test_return_all_samples(self, jaxns_bridge):
        """Test that return_all_samples returns correct structure."""
        result = jaxns_bridge.return_all_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_sampled_samples(self, jaxns_bridge):
        """Test that return_sampled_samples returns correct structure."""
        result = jaxns_bridge.return_sampled_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_logZ(self, jaxns_bridge):
        """Test that return_logZ returns correct structure."""
        result = jaxns_bridge.return_logZ()
        validate_logZ_dict(result)


# ==============================================================================
# Nessai Tests
# ==============================================================================

@pytest.mark.slow
class TestNessaiInterface:
    """Tests for the Nessai sampler interface."""
    
    @pytest.fixture
    def nessai_bridge(self, model, priors, temp_output_dir):
        """Create a Nessai bridge and run sampler."""
        from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
        bridge = DiscoveryNessaiBridge(model, priors)
        bridge.run_sampler(nlive=50, max_iteration=300, output=temp_output_dir)
        return bridge
    
    def test_return_all_samples(self, nessai_bridge):
        """Test that return_all_samples returns correct structure."""
        result = nessai_bridge.return_all_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_sampled_samples(self, nessai_bridge):
        """Test that return_sampled_samples returns correct structure."""
        result = nessai_bridge.return_sampled_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_logZ(self, nessai_bridge):
        """Test that return_logZ returns correct structure."""
        result = nessai_bridge.return_logZ()
        validate_logZ_dict(result)


# ==============================================================================
# Eryn Tests
# ==============================================================================

@pytest.mark.slow
class TestErynInterface:
    """Tests for the Eryn sampler interface."""
    
    @pytest.fixture
    def eryn_bridge(self, model, priors):
        """Create an Eryn bridge and run sampler."""
        from discoverysamplers.eryn_interface import DiscoveryErynBridge
        bridge = DiscoveryErynBridge(model, priors)
        bridge.create_sampler(nwalkers=8)
        bridge.run_sampler(nsteps=200)
        return bridge
    
    def test_return_all_samples(self, eryn_bridge):
        """Test that return_all_samples returns correct structure."""
        result = eryn_bridge.return_all_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_sampled_samples(self, eryn_bridge):
        """Test that return_sampled_samples returns correct structure."""
        result = eryn_bridge.return_sampled_samples()
        validate_samples_dict(result)
        assert len(result['names']) == 2  # x and y
    
    def test_return_logZ_raises_error(self, eryn_bridge):
        """Test that return_logZ raises NotImplementedError for MCMC."""
        with pytest.raises(NotImplementedError):
            eryn_bridge.return_logZ()


# ==============================================================================
# Cross-interface consistency tests
# ==============================================================================

@pytest.mark.slow
class TestCrossInterfaceConsistency:
    """Tests that verify consistency across different sampler interfaces."""
    
    def test_return_all_samples_keys_match(self, model, priors, temp_output_dir):
        """Test that all samplers return the same keys in return_all_samples."""
        from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
        from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
        from discoverysamplers.eryn_interface import DiscoveryErynBridge
        
        # JAX-NS
        jaxns = DiscoveryJAXNSBridge(model, priors)
        jaxns.run_sampler(nlive=50, max_samples=500)
        jaxns_result = jaxns.return_all_samples()
        
        # Nessai
        nessai = DiscoveryNessaiBridge(model, priors)
        nessai.run_sampler(nlive=50, max_iteration=300, output=temp_output_dir)
        nessai_result = nessai.return_all_samples()
        
        # Eryn
        eryn = DiscoveryErynBridge(model, priors)
        eryn.create_sampler(nwalkers=8)
        eryn.run_sampler(nsteps=200)
        eryn_result = eryn.return_all_samples()
        
        # Compare keys
        assert set(jaxns_result.keys()) == set(nessai_result.keys())
        assert set(jaxns_result.keys()) == set(eryn_result.keys())
    
    def test_return_logZ_keys_match(self, model, priors, temp_output_dir):
        """Test that nested samplers return the same keys in return_logZ."""
        from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
        from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
        
        # JAX-NS
        jaxns = DiscoveryJAXNSBridge(model, priors)
        jaxns.run_sampler(nlive=50, max_samples=500)
        jaxns_logZ = jaxns.return_logZ()
        
        # Nessai
        nessai = DiscoveryNessaiBridge(model, priors)
        nessai.run_sampler(nlive=50, max_iteration=300, output=temp_output_dir)
        nessai_logZ = nessai.return_logZ()
        
        # Compare keys
        assert set(jaxns_logZ.keys()) == set(nessai_logZ.keys())


# ==============================================================================
# Run tests if executed directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
