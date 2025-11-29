# discoverysamplers

**Bridge interfaces connecting Discovery models to various sampling algorithms.**

[![Documentation Status](https://readthedocs.org/projects/discoverysamplers/badge/?version=latest)](https://discoverysamplers.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/jonaselgammal/discoverysamplers/actions/workflows/tests.yml/badge.svg)](https://github.com/jonaselgammal/discoverysamplers/actions/workflows/tests.yml)

📖 **Documentation**: [https://discoverysamplers.readthedocs.io](https://discoverysamplers.readthedocs.io)

## Overview

`discoverysamplers` provides lightweight wrappers that adapt Discovery-style models (callables accepting parameter dictionaries) to the APIs expected by different sampling backends.

### Supported Samplers

- **Eryn**: Ensemble MCMC with parallel tempering and reversible-jump support
- **Nessai**: Flow-based nested sampling with importance sampling
- **JAX-NS**: Pure JAX nested sampling with GPU support
- **GPry**: Gaussian process emulation via Cobaya framework

## Installation

```bash
git clone https://github.com/jonaselgammal/discoverysamplers.git
cd discoverysamplers
pip install .
```

## Quick Example

```python
from discoverysamplers import DiscoveryNessaiBridge

# Define model and priors
def model(params):
    return -0.5 * (params['x']**2 + params['y']**2)

priors = {
    'x': ('uniform', -5.0, 5.0),
    'y': ('uniform', -5.0, 5.0),
}

# Run sampler
bridge = DiscoveryNessaiBridge(model, priors)
results = bridge.run_sampler(nlive=1000, output='output/')
```

## Usage

### MCMC with Eryn

```python
from discoverysamplers import DiscoveryErynBridge

bridge = DiscoveryErynBridge(model, priors)
bridge.create_sampler(nwalkers=32)
bridge.run_sampler(nsteps=10000)
samples = bridge.return_all_samples()
```

### Nested Sampling with JAX-NS

```python
from discoverysamplers import DiscoveryJAXNSBridge

bridge = DiscoveryJAXNSBridge(model, priors)
results = bridge.run_sampler(nlive=1000, rng_seed=42)
```

## Prior Specification

```python
priors = {
    'mass': ('uniform', 1.0, 3.0),
    'distance': ('loguniform', 0.1, 100.0),
    'phase': ('normal', 0.0, 1.0),
    'fixed_param': ('fixed', 1.0),
}
```

## Documentation

Full documentation: [https://discoverysamplers.readthedocs.io](https://discoverysamplers.readthedocs.io)

## Citation

```bibtex
@software{discoverysamplers,
  author = {El Gammal, Jonas},
  title = {discoverysamplers: Bridge interfaces for Bayesian samplers},
  year = {2024},
  url = {https://github.com/jonaselgammal/discoverysamplers}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
