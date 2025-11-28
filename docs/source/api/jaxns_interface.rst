JAX-NS Interface API
====================

.. automodule:: discoverysamplers.jaxns_interface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

DiscoveryJAXNSBridge
--------------------

.. autoclass:: discoverysamplers.jaxns_interface.DiscoveryJAXNSBridge
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiscoveryJAXNSBridge.configure_array_api
      ~DiscoveryJAXNSBridge.run_sampler
      ~DiscoveryJAXNSBridge.sample_priors

   .. rubric:: Attributes

   .. autosummary::

      ~DiscoveryJAXNSBridge.discovery_paramnames
      ~DiscoveryJAXNSBridge.sampled_names
      ~DiscoveryJAXNSBridge.fixed_names
      ~DiscoveryJAXNSBridge.n_sampled
      ~DiscoveryJAXNSBridge.n_fixed
      ~DiscoveryJAXNSBridge.sampled_prior_dict
      ~DiscoveryJAXNSBridge.fixed_param_dict
      ~DiscoveryJAXNSBridge.latex_labels

Helper Classes
--------------

ParsedPrior
^^^^^^^^^^^

.. autoclass:: discoverysamplers.jaxns_interface.ParsedPrior
   :members:
   :undoc-members:

Method Details
--------------

configure_array_api
^^^^^^^^^^^^^^^^^^^

.. automethod:: discoverysamplers.jaxns_interface.DiscoveryJAXNSBridge.configure_array_api

run_sampler
^^^^^^^^^^^

.. automethod:: discoverysamplers.jaxns_interface.DiscoveryJAXNSBridge.run_sampler

sample_priors
^^^^^^^^^^^^^

.. automethod:: discoverysamplers.jaxns_interface.DiscoveryJAXNSBridge.sample_priors

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
   import jax.numpy as jnp

   # Define JAX-compatible model
   def model(params):
       x = params['x']
       y = params['y']
       return -0.5 * (jnp.square(x) + jnp.square(y))

   # Define priors
   priors = {
       'x': ('uniform', -5, 5),
       'y': ('uniform', -5, 5),
   }

   # Create bridge
   bridge = DiscoveryJAXNSBridge(
       discovery_model=model,
       priors=priors,
       jit=True
   )

   # Run sampler
   results = bridge.run_sampler(
       nlive=1000,
       max_samples=10000,
       rng_seed=42
   )

   # Access results
   print(f"Log evidence: {results['logZ']}")

With Vectorization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax

   # Enable 64-bit precision
   jax.config.update("jax_enable_x64", True)

   # Create bridge
   bridge = DiscoveryJAXNSBridge(model, priors, jit=True)

   # Enable vectorized evaluation
   bridge.configure_array_api(order=['x', 'y'])

   # Run sampler (automatically uses vectorization)
   results = bridge.run_sampler(
       nlive=1000,
       max_samples=10000,
       rng_seed=42
   )

Custom Termination
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Run with custom termination criterion
   results = bridge.run_sampler(
       nlive=2000,
       max_samples=50000,
       termination_frac=0.001,  # Stop at 99.9% of evidence
       rng_seed=42
   )

GPU Acceleration
^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax

   # Check available devices
   print(jax.devices())

   # JAX-NS automatically uses GPU if available
   # No special configuration needed
   bridge = DiscoveryJAXNSBridge(model, priors, jit=True)
   results = bridge.run_sampler(nlive=1000, max_samples=10000)

Working with Results
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp

   # Extract samples
   samples = results['samples']  # Dictionary of JAX arrays
   weights = results['weights']   # Posterior weights

   x_samples = samples['x']
   y_samples = samples['y']

   # Compute statistics
   x_mean = jnp.average(x_samples, weights=weights)
   x_std = jnp.sqrt(jnp.average(jnp.square(x_samples - x_mean), weights=weights))

   print(f"x = {x_mean:.3f} ± {x_std:.3f}")

   # Evidence and information
   print(f"Log evidence: {results['logZ']:.2f} ± {results['logZerr']:.2f}")
   print(f"Information: {results['H']:.2f} nats")
   print(f"ESS: {results['ESS']:.0f}")

See Also
--------

- :doc:`../user_guide/jaxns_usage` - Usage guide
- :doc:`../advanced/performance` - Performance optimization
