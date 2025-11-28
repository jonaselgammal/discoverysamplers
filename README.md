# discoverysamplers

**A collection of bridge interfaces connecting Discovery models to various sampling algorithms.**

[![Documentation Status](https://readthedocs.org/projects/discoverysamplers/badge/?version=latest)](https://discoverysamplers.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`discoverysamplers` provides lightweight wrappers that adapt Discovery-style models (callables accepting parameter dictionaries) to the APIs expected by different sampling backends. This package enables seamless switching between samplers and consistent interface for Bayesian inference tasks.

### Supported Samplers

- **Eryn**: Ensemble MCMC with parallel tempering and reversible-jump support
- **Nessai**: Flow-based nested sampling with importance sampling
- **JAX-NS**: Pure JAX nested sampling with GPU support and vectorization
- **GPry**: Gaussian process sampler via Cobaya framework

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/discoverysamplers.git
cd discoverysamplers

# Install the package
pip install .

# Then install specific samplers as needed
pip install nessai  # For Nessai
pip install eryn>=1.2  # For Eryn
pip install jaxns jax  # For JAX-NS
pip install gpry cobaya  # For GPry
```

### Basic Example

```python
from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
import numpy as np

# Define your model
def gaussian_model(params):
    x = params['x']
    y = params['y']
    return -0.5 * (x**2 + y**2)

# Define priors
priors = {
    'x': ('uniform', -5.0, 5.0),
    'y': ('uniform', -5.0, 5.0),
}

# Create bridge and run sampler
bridge = DiscoveryNessaiBridge(gaussian_model, priors, jit=True)
results = bridge.run_sampler(nlive=1000, output='output/')

# Access results
print(f"Log evidence: {results['logZ']} ± {results['logZ_err']}")
```

## Key Features

- **Unified Interface**: Consistent API across multiple sampling backends
- **Flexible Prior Specification**: Multiple formats (dicts, tuples, callables)
- **JAX Integration**: Optional JIT compilation for performance
- **Fixed Parameters**: Automatic handling of fixed vs. sampled parameters
- **LaTeX Labels**: Built-in support for publication-quality plots

## Documentation

Full documentation available at: [https://discoverysamplers.readthedocs.io](https://discoverysamplers.readthedocs.io)

### Quick Links

- [Installation Guide](https://discoverysamplers.readthedocs.io/en/latest/user_guide/installation.html)
- [Quick Start Tutorial](https://discoverysamplers.readthedocs.io/en/latest/user_guide/quickstart.html)
- [API Reference](https://discoverysamplers.readthedocs.io/en/latest/api/eryn_interface.html)
- [Example Notebooks](https://discoverysamplers.readthedocs.io/en/latest/examples/notebooks.html)

## Usage Examples

### MCMC with Eryn

```python
from discoverysamplers.eryn_interface import DiscoveryErynBridge

bridge = DiscoveryErynBridge(model, priors)
sampler = bridge.create_sampler(nwalkers=32)

# Initialize and run
initial = bridge.sample_priors(nwalkers=32)
sampler.run_mcmc(initial, nsteps=10000)

# Get samples
samples = sampler.get_chain(discard=1000, flat=True)
```

### Nested Sampling with JAX-NS

```python
from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
import jax

jax.config.update("jax_enable_x64", True)

bridge = DiscoveryJAXNSBridge(jax_model, priors, jit=True)
bridge.configure_array_api(order=['x', 'y'])  # Enable vectorization

results = bridge.run_sampler(nlive=1000, max_samples=10000, rng_seed=42)
```

### Parallel Tempering

```python
# Handle multimodal posteriors with parallel tempering
sampler = bridge.create_sampler(
    nwalkers=32,
    ntemps=8,
    Tmax=20.0
)

initial = bridge.sample_priors(nwalkers=32, ntemps=8)
sampler.run_mcmc(initial, nsteps=10000)

# Get cold chain samples
samples = sampler.get_chain(discard=1000, flat=True, temp=0)
```

## Prior Specification

Multiple formats supported:

```python
# Tuple format (shorthand)
priors = {
    'mass': ('uniform', 1.0, 3.0),
    'distance': ('loguniform', 0.1, 100.0),
    'phase': ('normal', 0.0, 1.0),
    'fixed_param': ('fixed', 1.0),
}

# Dictionary format
priors = {
    'mass': {'dist': 'uniform', 'min': 1.0, 'max': 3.0},
    'distance': {'dist': 'loguniform', 'min': 0.1, 'max': 100.0},
}

# Custom callable priors
class CustomPrior:
    def __init__(self, a, b):
        self.bounds = (a, b)

    def logpdf(self, value):
        if self.bounds[0] <= value <= self.bounds[1]:
            return -np.log(self.bounds[1] - self.bounds[0])
        return -np.inf

priors = {'param': CustomPrior(0, 10)}
```

## Requirements

### Core
- Python >= 3.9
- NumPy

### Optional (sampler-specific)
- **eryn** >= 1.2 - For Eryn interface
- **nessai** - For Nessai interface
- **jaxns** - For JAX-NS interface
- **gpry** - For GPry interface
- **cobaya** - Required for GPry interface
- **jax** - Recommended for performance (used by JAX-NS)

## Development

### Setting Up

```bash
git clone https://github.com/yourusername/discoverysamplers.git
cd discoverysamplers
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Building Documentation

```bash
cd docs/
make html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Ways to contribute:
- Report bugs or request features via [GitHub Issues](https://github.com/yourusername/discoverysamplers/issues)
- Submit pull requests
- Improve documentation
- Add example notebooks

## Citation

If you use `discoverysamplers` in your research, please cite:

```bibtex
@software{discoverysamplers,
  author = {El Gammal, Jonas},
  title = {discoverysamplers: Bridge interfaces for Bayesian samplers},
  year = {2024},
  url = {https://github.com/yourusername/discoverysamplers}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

`discoverysamplers` builds upon excellent work from:
- [Eryn](https://github.com/mikekatz04/Eryn) by Michael Katz
- [Nessai](https://github.com/mj-will/nessai) by Michael Williams
- [JAX-NS](https://github.com/Joshuaalbert/jaxns) by Joshua Albert
- [GPry](https://github.com/GuillermoFrancoAbellan/GPry) by Guillermo Franco
- [Cobaya](https://github.com/CobayaSampler/cobaya) by the Cobaya team
- [Discovery](https://github.com/ark0015/discovery) PTA analysis framework

## Support

- **Documentation**: [https://discoverysamplers.readthedocs.io](https://discoverysamplers.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/discoverysamplers/issues)
- **Examples**: See `examples/` directory

## Version

Current version: 0.1.0 (development)
