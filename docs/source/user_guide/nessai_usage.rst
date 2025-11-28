Nessai Nested Sampler
=====================

Interface to `Nessai <https://nessai.readthedocs.io/>`_, a nested sampler that trains normalizing flows for proposals. Use it when you need evidence estimates or robust exploration of multi-modal posteriors.

Minimal Run
-----------

.. code-block:: python

   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
   import numpy as np

   # Define your model
   def my_model(params):
       x = params['x']
       y = params['y']
       return -0.5 * (x**2 + y**2)

   # Define priors
   priors = {
       'x': ('uniform', -5.0, 5.0),
       'y': ('uniform', -5.0, 5.0),
   }

   # Create the bridge (accepts callable or object with .logL attribute)
   bridge = DiscoveryNessaiBridge(
       discovery_model=my_model,
       priors=priors,
       latex_labels={'x': r'$x$', 'y': r'$y$'},
       jit=True  # Optional JAX JIT for the model
   )

   # Run nested sampling (key options shown)
   results = bridge.run_sampler(
       nlive=1000,                  # Live points (accuracy vs cost)
       output='output/nessai_run/', # Checkpoints & diagnostics
       resume=False,                # Resume if previous run exists
   )

   # Evidence and samples
   print(f"Log evidence: {results['logZ']} ± {results['logZ_err']}")
   print(f"Information: {results['information']} nats")
   posterior_samples = results['posterior_samples']

Using with Discovery Likelihoods
--------------------------------

.. code-block:: python

   import discovery as ds

   # Create Discovery likelihood
   psr = ds.Pulsar.read_feather('path/to/pulsar.feather')
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),
   ])

   # Pass the likelihood object directly (bridge extracts .logL automatically)
   bridge = DiscoveryNessaiBridge(
       discovery_model=likelihood,  # or likelihood.logL - both work
       priors=priors,
       jit=True
   )

Important Options
------------------

.. code-block:: python

   results = bridge.run_sampler(
       nlive=1000,                 # ↑ accuracy, ↓ speed
       max_iteration=10000,        # Stop after this many iterations
       resume=True,                # Continue from checkpoints
       flow_config={
           'model_config': {'n_blocks': 6, 'n_neurons': 64},
           'training': {'max_epochs': 50, 'patience': 5},
       },
       n_pool=1000,                # Proposal pool size
   )

Reading Results
---------------

.. code-block:: python

   posterior = results['posterior_samples']   # structured array
   weights = posterior['weights']
   x_mean = np.average(posterior['x'], weights=weights)
   x_std = np.sqrt(np.average((posterior['x'] - x_mean)**2, weights=weights))

Basic tips:
- Start with ``nlive=500-1000`` and increase if ``logZ_err`` is too large.
- Keep priors bounded (required); check ``diagnostics/`` in the output folder.
- Use ``resume=True`` to continue partially finished runs.

See Also
--------

- :doc:`../api/nessai_interface` - Complete API reference
- :doc:`../advanced/performance` - Performance optimization
- `Nessai documentation <https://nessai.readthedocs.io/>`_ - Official Nessai docs
