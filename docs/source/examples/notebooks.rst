Example Notebooks
=================

This page provides links to example Jupyter notebooks demonstrating various use cases of ``discoverysamplers``.

Available Examples
------------------

The ``examples/`` directory contains Jupyter notebooks showcasing different features and samplers:

Quick Start Guide
^^^^^^^^^^^^^^^^^

**quick_start_guide.ipynb**

Comprehensive introduction to all samplers:

- Basic model setup
- Using all four samplers (Nessai, JAX-NS, Eryn, GPry)
- Parallel tempering with Eryn
- Results analysis and visualization

.. code-block:: bash

   # Location
   examples/quick_start_guide.ipynb

Reversible Jump MCMC
^^^^^^^^^^^^^^^^^^^^

**RJ_MCMC.ipynb**

Trans-dimensional sampling with Eryn's RJMCMC:

- Toy example with variable number of Gaussian components
- Discovery PTA example with continuous wave sources
- Branch-indexed prior specification
- Component count analysis and plotting

.. code-block:: bash

   # Location
   examples/RJ_MCMC.ipynb

Eryn Examples
^^^^^^^^^^^^^

**eryn_example.ipynb**

Basic usage of the Eryn MCMC sampler:

- Setting up a simple model
- Configuring priors
- Running MCMC sampling
- Analyzing chains and convergence
- Visualizing results

.. code-block:: bash

   # Location
   examples/eryn_example.ipynb

Nessai Examples
^^^^^^^^^^^^^^^

**nessai_example.ipynb**

Nested sampling with Nessai:

- Flow-based nested sampling
- Configuring flow parameters
- Evidence calculation
- Posterior sampling
- Model comparison

.. code-block:: bash

   # Location
   examples/nessai_example.ipynb

Running the Examples
--------------------

Prerequisites
^^^^^^^^^^^^^

Install required dependencies:

.. code-block:: bash

   # Install discoverysamplers with all samplers
   pip install discoverysamplers eryn nessai jaxns gpry cobaya jax

   # Install Jupyter
   pip install jupyter matplotlib corner

Running Notebooks
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Navigate to examples directory
   cd examples/

   # Start Jupyter
   jupyter notebook

   # Open desired notebook in browser

Converting to Python Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run examples as scripts:

.. code-block:: bash

   # Convert notebook to Python script
   jupyter nbconvert --to python eryn_example.ipynb

   # Run the script
   python eryn_example.py

Example Snippets
----------------

Quick Start Example
^^^^^^^^^^^^^^^^^^^

A minimal working example:

.. code-block:: python

   import numpy as np
   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

   # Define a simple 2D Gaussian model
   def gaussian_model(params):
       x = params['x']
       y = params['y']
       return -0.5 * (x**2 + y**2)

   # Define priors
   priors = {
       'x': ('uniform', -5, 5),
       'y': ('uniform', -5, 5),
   }

   # Create bridge
   bridge = DiscoveryNessaiBridge(
       discovery_model=gaussian_model,
       priors=priors,
       jit=True
   )

   # Run sampler
   results = bridge.run_sampler(
       nlive=1000,
       output='output/gaussian/',
       resume=False
   )

   # Print results
   print(f"Log evidence: {results['logZ']:.2f} ± {results['logZ_err']:.2f}")

Multimodal Example
^^^^^^^^^^^^^^^^^^

Handling multimodal distributions:

.. code-block:: python

   from scipy.special import logsumexp
   from discoverysamplers.eryn_interface import DiscoveryErynBridge

   # Bimodal likelihood
   def bimodal_model(params):
       x = params['x']
       y = params['y']

       # Two modes at (-2, -2) and (2, 2)
       log_L1 = -0.5 * ((x + 2)**2 + (y + 2)**2)
       log_L2 = -0.5 * ((x - 2)**2 + (y - 2)**2)

       return logsumexp([log_L1, log_L2]) - np.log(2)

   priors = {
       'x': ('uniform', -6, 6),
       'y': ('uniform', -6, 6),
   }

   # Use parallel tempering for multimodality
   bridge = DiscoveryErynBridge(bimodal_model, priors)

   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=8, Tmax=20.0)
   )

   initial = bridge.sample_priors(nwalkers=32, ntemps=8)
   sampler.run_mcmc(initial, nsteps=10000)

   # Get cold chain samples
   samples = sampler.get_chain(discard=1000, flat=True)

High-Dimensional Example
^^^^^^^^^^^^^^^^^^^^^^^^^

Sampling in higher dimensions with JAX-NS:

.. code-block:: python

   import jax.numpy as jnp
   from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge

   # 10-dimensional Gaussian
   ndim = 10

   def high_dim_model(params):
       x = jnp.array([params[f'x{i}'] for i in range(ndim)])
       return -0.5 * jnp.sum(x**2)

   priors = {f'x{i}': ('uniform', -5, 5) for i in range(ndim)}

   bridge = DiscoveryJAXNSBridge(high_dim_model, priors, jit=True)

   # Enable vectorization for speed
   bridge.configure_array_api(order=[f'x{i}' for i in range(ndim)])

   results = bridge.run_sampler(
       nlive=1000,
       max_samples=20000,
       rng_seed=42
   )

Realistic Science Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

Fitting a sinusoidal signal:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Generate synthetic data
   np.random.seed(42)
   times = np.linspace(0, 10, 100)

   true_params = {
       'amplitude': 2.5,
       'frequency': 1.5,
       'phase': 0.3,
       'noise_std': 0.5,
   }

   signal = (true_params['amplitude'] *
            np.sin(2*np.pi*true_params['frequency']*times + true_params['phase']))
   noise = np.random.normal(0, true_params['noise_std'], len(times))
   data = signal + noise

   # Define model
   def sinusoid_model(params):
       A = params['amplitude']
       f = params['frequency']
       phi = params['phase']
       sigma = params['noise_std']

       model_signal = A * np.sin(2*np.pi*f*times + phi)
       residuals = data - model_signal

       # Gaussian likelihood
       log_L = -0.5 * np.sum((residuals/sigma)**2) - len(data)*np.log(sigma)
       return log_L

   # Priors
   priors = {
       'amplitude': ('uniform', 0, 5),
       'frequency': ('uniform', 0.1, 3),
       'phase': ('uniform', 0, 2*np.pi),
       'noise_std': ('loguniform', 0.01, 2),
   }

   # Run sampling
   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

   bridge = DiscoveryNessaiBridge(sinusoid_model, priors, jit=False)
   results = bridge.run_sampler(nlive=1000, output='output/sinusoid/')

   # Analyze results
   posterior = results['posterior_samples']
   weights = posterior['weights']

   for param in ['amplitude', 'frequency', 'phase', 'noise_std']:
       samples = posterior[param]
       mean = np.average(samples, weights=weights)
       std = np.sqrt(np.average((samples - mean)**2, weights=weights))
       true_val = true_params[param]

       print(f"{param}:")
       print(f"  True: {true_val:.3f}")
       print(f"  Estimated: {mean:.3f} ± {std:.3f}")

Visualization Examples
----------------------

Corner Plots
^^^^^^^^^^^^

.. code-block:: python

   import corner
   import numpy as np

   # Get samples
   posterior = results['posterior_samples']
   samples_array = np.column_stack([
       posterior[name] for name in bridge.sampled_names
   ])
   weights = posterior['weights']

   # Create corner plot
   fig = corner.corner(
       samples_array,
       weights=weights,
       labels=[bridge.latex_labels.get(n, n) for n in bridge.sampled_names],
       quantiles=[0.16, 0.5, 0.84],
       show_titles=True,
       title_kwargs={"fontsize": 12},
   )

   plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')

Chain Diagnostics
^^^^^^^^^^^^^^^^^

For MCMC (Eryn):

.. code-block:: python

   # Get chain
   chain = sampler.get_chain()  # Shape: (nsteps, nwalkers, ndim)

   # Plot traces
   fig, axes = plt.subplots(bridge.ndim, figsize=(10, 2*bridge.ndim))

   for i, name in enumerate(bridge.sampled_names):
       ax = axes[i] if bridge.ndim > 1 else axes
       for walker in range(chain.shape[1]):
           ax.plot(chain[:, walker, i], alpha=0.3)
       ax.set_ylabel(bridge.latex_labels.get(name, name))
       ax.set_xlabel('Step')

   plt.tight_layout()
   plt.savefig('traces.png')

Model Predictions
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Generate predictions from posterior
   n_posterior_samples = 100
   predictions = []

   indices = np.random.choice(
       len(posterior['weights']),
       size=n_posterior_samples,
       p=posterior['weights']/posterior['weights'].sum()
   )

   for idx in indices:
       params = {name: posterior[name][idx] for name in bridge.sampled_names}
       pred = generate_signal(params)  # Your signal generation
       predictions.append(pred)

   predictions = np.array(predictions)

   # Plot with credible intervals
   plt.figure(figsize=(12, 6))
   plt.plot(times, data, 'k.', label='Data', alpha=0.5)

   median = np.median(predictions, axis=0)
   lower = np.percentile(predictions, 16, axis=0)
   upper = np.percentile(predictions, 84, axis=0)

   plt.plot(times, median, 'r-', label='Median', lw=2)
   plt.fill_between(times, lower, upper, alpha=0.3, label='68% CI')

   plt.xlabel('Time')
   plt.ylabel('Signal')
   plt.legend()
   plt.savefig('predictions.png')

Additional Resources
--------------------

External Examples
^^^^^^^^^^^^^^^^^

- `Eryn examples <https://github.com/mikekatz04/Eryn/tree/main/examples>`_
- `Nessai examples <https://github.com/mj-will/nessai/tree/main/examples>`_
- `JAX-NS examples <https://github.com/Joshuaalbert/jaxns/tree/main/examples>`_

Tutorials
^^^^^^^^^

- :doc:`../user_guide/quickstart` - Quick start guide
- :doc:`../user_guide/eryn_usage` - Eryn detailed usage
- :doc:`../user_guide/nessai_usage` - Nessai detailed usage
- :doc:`../user_guide/jaxns_usage` - JAX-NS detailed usage

See Also
--------

- :doc:`../user_guide/model_requirements` - Model requirements
- :doc:`../user_guide/prior_specification` - Prior specifications
- :doc:`../advanced/performance` - Performance tips
