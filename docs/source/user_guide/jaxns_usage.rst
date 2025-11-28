JAX-NS Nested Sampler
=====================

Interface to `JAX-NS <https://jaxns.readthedocs.io/>`_, a pure-JAX nested sampler. Use it for JAX-native models, vectorized likelihoods, and GPU/TPU acceleration.

Minimal Run
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge

   jax.config.update("jax_enable_x64", True)

   def my_model(params):
       x, y = params['x'], params['y']
       return -0.5 * (jnp.square(x) + jnp.square(y))

   priors = {'x': ('uniform', -5.0, 5.0), 'y': ('uniform', -5.0, 5.0)}

   # Create the bridge (accepts callable or object with .logL attribute)
   bridge = DiscoveryJAXNSBridge(
       discovery_model=my_model,
       priors=priors,
       latex_labels={'x': r'$x$', 'y': r'$y$'},
       jit=True,
   )

   results = bridge.run_sampler(
       nlive=800,
       max_samples=10000,
       termination_frac=0.01,
       rng_seed=42,
   )

   print(f"logZ = {results['logZ']} ± {results['logZerr']}")

Using with Discovery Likelihoods
--------------------------------

.. code-block:: python

   import discovery as ds

   # Create Discovery likelihood
   psr = ds.Pulsar.read_feather('path/to/pulsar.feather')
   likelihood = ds.PulsarLikelihood([...])

   # Pass the likelihood object directly
   bridge = DiscoveryJAXNSBridge(
       discovery_model=likelihood,  # or likelihood.logL - both work
       priors=priors,
       jit=True
   )

Vectorized Evaluation
---------------------

Enable batching when your model supports array inputs:

.. code-block:: python

   bridge.configure_array_api(order=['x', 'y'])  # parameter order for arrays
   results = bridge.run_sampler(nlive=800, max_samples=8000)

Key Options
-----------

- ``nlive``: accuracy vs. cost (start 500–1000).
- ``max_samples``: hard cap; set high enough to reach termination.
- ``termination_frac``: smaller = more accurate evidence.
- ``jit``: keep True for JAX models; disable for pure NumPy.
- Priors must be uniform/loguniform/normal/fixed; callable priors are not supported by the JAX-NS bridge.

Reading Results
---------------

.. code-block:: python

   samples = results['samples']
   weights = results['weights']
   x_mean = jnp.average(samples['x'], weights=weights)
   x_std = jnp.sqrt(jnp.average(jnp.square(samples['x'] - x_mean), weights=weights))

Tips
----

- Always enable x64 for nested sampling precision.
- Keep priors bounded when possible for stable transforms.
- Monitor ``results['ESS']`` and ``results['logZerr']`` to judge run quality.
- Use GPU automatically if available (``jax.devices()``).

See Also
--------

- :doc:`../api/jaxns_interface` - API reference
- :doc:`../advanced/performance` - Performance optimization
- `JAX documentation <https://jax.readthedocs.io/>`_ - JAX ecosystem docs
