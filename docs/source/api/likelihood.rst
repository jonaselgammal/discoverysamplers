Likelihood Module API
======================

.. automodule:: discoverysamplers.likelihood
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``likelihood`` module provides model adapter utilities for wrapping Discovery
models to work with various sampler backends. This module handles JIT compilation,
fixed parameter injection, and vectorized evaluation.

Classes
-------

LikelihoodWrapper
^^^^^^^^^^^^^^^^^

.. autoclass:: discoverysamplers.likelihood.LikelihoodWrapper
   :members:
   :undoc-members:
   :show-inheritance:

   Wrapper class for adapting Discovery models with:

   - **JAX JIT compilation** support for improved performance
   - **Fixed parameter injection** to automatically merge fixed and sampled parameters
   - **Vectorized evaluation** (array API) for batch likelihood computations
   - Three evaluation modes:

     - ``log_likelihood(params_dict)`` - Single parameter set evaluation
     - ``log_likelihood_row(params_dict)`` - Batched evaluation with dict of arrays
     - ``log_likelihood_matrix(params_array)`` - Batched evaluation with 2D array

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.likelihood import LikelihoodWrapper
   import discovery as ds

   # Create Discovery likelihood
   psr = ds.Pulsar.read_feather('pulsar.feather')
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict)
   ])

   # Wrap with adapter
   fixed_params = {'param1': 1.0}
   adapter = LikelihoodWrapper(
       model=likelihood.logL,
       fixed_params=fixed_params,
       jit=True
   )

   # Evaluate likelihood (fixed params auto-injected)
   params = {'param2': 2.0, 'param3': 3.0}
   log_L = adapter.log_likelihood(params)

With JIT Compilation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Enable JIT for faster evaluation (default)
   adapter = LikelihoodWrapper(
       model=likelihood.logL,
       jit=True  # Compiles model with JAX JIT
   )

   # First call will compile, subsequent calls are fast
   log_L = adapter.log_likelihood(params)

Vectorized Evaluation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp

   # Enable array API for vectorized evaluation
   adapter = LikelihoodWrapper(
       model=likelihood.logL,
       allow_array_api=True,
       jit=True
   )

   # Configure parameter order for array API
   adapter.configure_array_api(['param1', 'param2', 'param3'])

   # Evaluate with dict of arrays (batched)
   params_batch = {
       'param1': jnp.array([1.0, 2.0, 3.0]),
       'param2': jnp.array([0.5, 1.0, 1.5]),
       'param3': jnp.array([0.1, 0.2, 0.3])
   }
   log_L_batch = adapter.log_likelihood_row(params_batch)
   # Returns array of shape (3,)

   # Or with 2D array
   params_array = jnp.array([
       [1.0, 0.5, 0.1],
       [2.0, 1.0, 0.2],
       [3.0, 1.5, 0.3]
   ])
   log_L_batch = adapter.log_likelihood_matrix(params_array)
   # Returns array of shape (3,)

Fixed Parameters
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Specify fixed parameters that will be auto-injected
   fixed_params = {
       'redshift': 0.2,
       'inclination': 1.57,
   }

   adapter = LikelihoodWrapper(
       model=likelihood.logL,
       fixed_params=fixed_params
   )

   # Only provide sampled parameters
   sampled_params = {
       'mass1': 1.4,
       'mass2': 1.3,
   }

   # Adapter automatically merges fixed_params + sampled_params
   log_L = adapter.log_likelihood(sampled_params)
   # Equivalent to: likelihood.logL({'mass1': 1.4, 'mass2': 1.3,
   #                                  'redshift': 0.2, 'inclination': 1.57})

See Also
--------

- :doc:`../user_guide/model_requirements` - Requirements for Discovery models
- :doc:`../advanced/performance` - Performance optimization tips
- :doc:`priors` - Prior specification utilities
