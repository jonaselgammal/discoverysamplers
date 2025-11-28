Quick Start Guide
=================

This guide will walk you through the basic usage of ``discoverysamplers`` with a simple example.

Basic Concepts
--------------

All sampler interfaces in ``discoverysamplers`` follow a common pattern:

1. **Define a model**: A callable that accepts a parameter dictionary and returns a log-likelihood
2. **Specify priors**: Define prior distributions for your parameters
3. **Create a bridge**: Instantiate the appropriate bridge class for your chosen sampler
4. **Run the sampler**: Execute the sampling algorithm
5. **Analyze results**: Process and visualize the output

Example: Discovery Pulsar Likelihood
-------------------------------------

Let's start with a Discovery likelihood for a single pulsar. Discovery likelihoods are the recommended way to use this package:

.. code-block:: python

   import numpy as np
   import jax
   jax.config.update('jax_enable_x64', True)

   import discovery as ds

   # Load pulsar data
   psr = ds.Pulsar.read_feather('path/to/pulsar.feather')

   # Create a PulsarLikelihood with measurement noise
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict)
   ])

   # The likelihood object has a logL attribute that's callable
   # and a params attribute containing parameter information
   print(f"Parameters: {likelihood.logL.params}")

**Note**: Discovery likelihood objects contain:
   - ``logL``: The callable log-likelihood function
   - ``logL.params``: Parameter specifications with bounds and priors

**Tip**: When creating a bridge, you can pass either ``likelihood`` or ``likelihood.logL`` — both work. The bridge automatically detects whether you passed the object or its ``logL`` callable.

Defining Priors
^^^^^^^^^^^^^^^

``discoverysamplers`` supports multiple prior specification formats. Here are three equivalent ways to specify uniform priors:

**Dictionary Format**:

.. code-block:: python

   priors = {
       'x': {'dist': 'uniform', 'min': -5.0, 'max': 5.0},
       'y': {'dist': 'uniform', 'min': -5.0, 'max': 5.0},
   }

**Tuple Format** (shorthand):

.. code-block:: python

   priors = {
       'x': ('uniform', -5.0, 5.0),
       'y': ('uniform', -5.0, 5.0),
   }

**Mixed Format**:

.. code-block:: python

   priors = {
       'x': ('uniform', -5.0, 5.0),
       'y': {'dist': 'loguniform', 'min': 0.1, 'max': 10.0},
       'fixed_param': ('fixed', 1.0),  # Fixed parameters
   }

See :doc:`prior_specification` for all supported prior types and formats.

Using Different Samplers
-------------------------

Nested Sampling with Nessai
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nessai is a flow-based nested sampling algorithm, ideal for multi-modal posteriors:

.. code-block:: python

   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
   import discovery as ds

   # Load pulsar and create likelihood
   psr = ds.Pulsar.read_feather('path/to/pulsar.feather')
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),
       ds.makegp_ecorr(psr, psr.noisedict),
       ds.makegp_timing(psr, svd=True)
   ])

   # Define priors (or use default from likelihood.logL.params)
   priors = {
       'param1': ('uniform', -5.0, 5.0),
       'param2': ('loguniform', 1e-10, 1e-5),
       # ... etc for your model parameters
   }

   # Create the bridge - you can pass likelihood or likelihood.logL
   bridge = DiscoveryNessaiBridge(
       discovery_model=likelihood,  # Pass the likelihood object (or likelihood.logL)
       priors=priors,
       jit=True  # Enable JAX JIT compilation
   )

   # Run the sampler
   results = bridge.run_sampler(
       nlive=1000,              # Number of live points
       output='output/nessai/', # Output directory
       resume=False,            # Start fresh
       max_iteration=10000      # Maximum iterations
   )

   # Access results
   print(f"Log evidence: {results['logZ']} +/- {results['logZ_err']}")
   samples = results['posterior_samples']  # Posterior samples

Nested Sampling with JAX-NS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JAX-NS is a pure JAX implementation with efficient vectorization:

.. code-block:: python

   from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
   import jax
   jax.config.update('jax_enable_x64', True)

   # Create the bridge with Discovery likelihood
   bridge = DiscoveryJAXNSBridge(
       discovery_model=likelihood,  # Pass the likelihood object (or likelihood.logL)
       priors=priors,
       jit=True
   )

   # Configure vectorized likelihood evaluation
   bridge.configure_array_api(order=['param1', 'param2'])

   # Run the sampler
   results = bridge.run_sampler(
       nlive=1000,
       max_samples=10000,
       termination_frac=0.01,
       rng_seed=42
   )

   # Results structure similar to Nessai
   print(f"Log evidence: {results['logZ']}")

MCMC with Eryn
^^^^^^^^^^^^^^

Eryn is an ensemble MCMC sampler (similar to emcee) with optional parallel tempering:

.. code-block:: python

   from discoverysamplers.eryn_interface import DiscoveryErynBridge

   # Create the bridge with Discovery likelihood
   bridge = DiscoveryErynBridge(
       model=likelihood,  # Pass the likelihood object (or likelihood.logL)
       priors=priors
   )

   # Create the sampler (basic MCMC)
   sampler = bridge.create_sampler(
       nwalkers=32,  # Number of walkers
   )

   # Generate initial positions from priors
   initial_state = bridge.sample_priors(nwalkers=32)

   # Run MCMC
   sampler.run_mcmc(
       initial_state,
       nsteps=10000,
       progress=True
   )

   # Get samples (discard burn-in)
   samples = sampler.get_chain(discard=1000, flat=True)
   log_probs = sampler.get_log_prob(discard=1000, flat=True)

**Parallel Tempering** (recommended for multi-modal posteriors):

.. code-block:: python

   # Create sampler with parallel tempering
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=8)  # 8 temperature chains
   )

   # Initial state must match shape (ntemps, nwalkers, ndim)
   initial_state = bridge.sample_priors(nwalkers=32, ntemps=8)

   # Run and extract cold chain (temp=0) samples
   sampler.run_mcmc(initial_state, nsteps=10000, progress=True)
   samples = sampler.get_chain(discard=1000, flat=True)  # Cold chain only

GPry via Cobaya
^^^^^^^^^^^^^^^

`GPry <https://gpry.readthedocs.io/en/latest/>`_ uses Gaussian Process emulation to accelerate inference for expensive likelihoods. It can be accessed through `Cobaya <https://cobaya.readthedocs.io/>`_:

.. code-block:: python

   from discoverysamplers.gpry_interface import DiscoveryGPryCobayaBridge

   # Create the bridge with Discovery likelihood
   bridge = DiscoveryGPryCobayaBridge(
       discovery_model=likelihood,  # Pass the likelihood object (or likelihood.logL)
       priors=priors,
       like_name='pulsar_likelihood'
   )

   # Run sampler
   updated_info, sampler = bridge.run_sampler(
       max_samples=10000,
       # Additional GPry/Cobaya options
   )

   # Access results through Cobaya
   products = sampler.products()
   samples = products["sample"]

Working with Fixed Parameters
------------------------------

Often you'll want to fix certain parameters while sampling others. With Discovery likelihoods, you can either:

1. **Use noise dictionary values**: Parameters specified in ``psr.noisedict`` are automatically fixed
2. **Override with prior specifications**: Define which parameters to sample vs. fix

.. code-block:: python

   import discovery as ds

   # Create likelihood with some parameters from noisedict (fixed)
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),  # Uses fixed values from noisedict
       ds.makegp_timing(psr, svd=True)  # Timing parameters will be sampled
   ])

   # Define priors - only for parameters you want to sample
   # Any parameters not in priors will use default bounds from likelihood.logL.params
   priors = {
       'timing_param1': ('uniform', -1e-7, 1e-7),
       'timing_param2': ('uniform', -1e-7, 1e-7),
       # Other parameters will use defaults or be fixed
   }

   # The bridge automatically handles parameter management
   bridge = DiscoveryNessaiBridge(likelihood.logL, priors)
   results = bridge.run_sampler(nlive=500, output='output/')

Fixed parameters are automatically separated and injected into the model during sampling.

Model Requirements
------------------

When using Discovery likelihoods:

1. **Use the likelihood object**: Pass ``likelihood`` (or ``likelihood.logL``) to the bridge
2. **Parameter dictionary**: The logL callable accepts a dict mapping parameter names to values
3. **Returns log-likelihood**: Returns a scalar log-likelihood value
4. **JAX compatible**: Discovery likelihoods are JAX-compatible and can be JIT-compiled

.. code-block:: python

   import discovery as ds

   # Create Discovery likelihood
   likelihood = ds.PulsarLikelihood([...])

   # The logL attribute is the callable likelihood function
   print(type(likelihood.logL))  # <class 'function'>

   # It accepts parameter dictionaries
   params = ds.sample_uniform(likelihood.logL.params)
   log_like = likelihood.logL(params)
   print(log_like)  # Returns a scalar

   # Pass likelihood object to the bridge (or likelihood.logL — both work)
   bridge = DiscoveryNessaiBridge(
       discovery_model=likelihood,  # The bridge extracts logL automatically
       priors=priors
   )

**Alternative**: You can also use plain Python functions (see :doc:`model_requirements` for details), but Discovery likelihoods are recommended for PTA analysis.

Next Steps
----------

Now that you understand the basics, explore:

- :doc:`prior_specification` - Complete guide to prior specifications
- :doc:`eryn_usage` - Advanced MCMC features (parallel tempering, RJMCMC)
- :doc:`nessai_usage` - Detailed Nessai configuration
- :doc:`jaxns_usage` - JAX-NS performance tuning
- :doc:`gpry_usage` - GPry and Cobaya options
- :doc:`../advanced/performance` - Performance optimization tips

Or check out the :doc:`../examples/notebooks` for real-world examples.
