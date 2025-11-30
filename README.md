# discoverysamplers

**Bridge interfaces connecting [Discovery](https://github.com/ark0015/discovery) models to various sampling algorithms.**

[![Documentation Status](https://readthedocs.org/projects/discoverysamplers/badge/?version=latest)](https://discoverysamplers.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/jonaselgammal/discoverysamplers/actions/workflows/tests.yml/badge.svg)](https://github.com/jonaselgammal/discoverysamplers/actions/workflows/tests.yml)

| | |
|---|---|
| **Author** | Jonas El Gammal |
| **Source** | [Source code on GitHub](https://github.com/jonaselgammal/discoverysamplers) |
| **Documentation** | [Documentation on Read the Docs](https://discoverysamplers.readthedocs.io) |
| **License** | [MIT](https://opensource.org/licenses/MIT) |
| **Support** | For questions, use [GitHub Issues](https://github.com/jonaselgammal/discoverysamplers/issues) or drop me an email |
| **Installation** | Clone from [GitHub](https://github.com/jonaselgammal/discoverysamplers) |

## Overview

`discoverysamplers` provides lightweight wrappers that adapt [Discovery](https://github.com/nanograv/discovery)-style models to the APIs expected by different sampling backends.

### Supported Samplers

| Sampler | Type | Description |
|---------|------|-------------|
| [Eryn](https://github.com/mikekatz04/Eryn) | MCMC | Ensemble MCMC with parallel tempering and reversible-jump support |
| [Nessai](https://github.com/mj-will/nessai) | Nested Sampling | Flow-based nested sampling with importance sampling |
| [JAX-NS](https://github.com/Joshuaalbert/jaxns) | Nested Sampling | Pure JAX nested sampling with GPU support |
| [GPry](https://github.com/jonaselgammal/GPry) | GP Emulation | Gaussian process surrogate model via Cobaya framework |

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

## Citation

```bibtex
@software{discoverysamplers,
  author = {El Gammal, Jonas},
  title = {discoverysamplers: Tools for Bayesian inference with Discovery},
  year = {2025},
  url = {https://github.com/jonaselgammal/discoverysamplers}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
