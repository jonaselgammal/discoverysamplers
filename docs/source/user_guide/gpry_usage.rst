GPry via Cobaya
===============

`GPry <https://gpry.readthedocs.io/en/latest/>`_ is a package for Bayesian inference of expensive likelihoods using Gaussian Process (GP) emulation. Instead of evaluating your likelihood thousands of times, GPry builds a surrogate GP model using active learning—intelligently choosing evaluation points to maximise information gain—and can reduce the number of likelihood evaluations by a factor of 100 or more. It is best suited for smooth, low-dimensional (≲20 parameters) likelihoods where each evaluation is computationally expensive.

This interface bridges Discovery models to GPry through `Cobaya <https://cobaya.readthedocs.io/>`_. It builds a Cobaya model from your Discovery likelihood and priors, then runs GPry (or another Cobaya sampler) on it.

Minimal Run
-----------

.. code-block:: python

   from discoverysamplers.gpry_interface import DiscoveryGPryCobayaBridge

   def my_model(params):
       x, y = params['x'], params['y']
       return -0.5 * (x**2 + y**2)

   priors = {'x': ('uniform', -5.0, 5.0), 'y': ('uniform', -5.0, 5.0)}

   # Create the bridge (accepts callable or object with .logL attribute)
   bridge = DiscoveryGPryCobayaBridge(
       discovery_model=my_model,
       priors=priors,
       latex_labels={'x': r'$x$', 'y': r'$y$'},
       like_name='my_like',
   )

   info, sampler = bridge.run_sampler(max_samples=5000)  # defaults to GPry
   products = sampler.products()
   samples = products["sample"]
   print(samples.mean('x'), samples.std('x'))

Using with Discovery Likelihoods
--------------------------------

.. code-block:: python

   import discovery as ds

   # Create Discovery likelihood
   psr = ds.Pulsar.read_feather('path/to/pulsar.feather')
   likelihood = ds.PulsarLikelihood([...])

   # Pass the likelihood object directly
   bridge = DiscoveryGPryCobayaBridge(
       discovery_model=likelihood,  # or likelihood.logL - both work
       priors=priors,
       like_name='pulsar_likelihood'
   )

Priors
------

- Uniform: ``("uniform", min, max)`` → Cobaya ``prior: {min, max}``
- Log-uniform: ``("loguniform", a, b)`` → Cobaya ``prior: {dist: loguniform, a, b}``
- Normal: ``("normal", mean, sigma[, min, max])`` → Cobaya ``prior: {dist: norm, loc, scale}``
- Fixed: ``("fixed", value)`` → fixed parameter
- Callable priors are **not** supported here.

Key Options
-----------

- ``sampler``: choose Cobaya sampler (``'gpry'`` default, or ``'mcmc'``, ``'polychord'`` if installed).
- ``max_samples``: total samples for GPry.
- ``alternative_paramnames``: remap model names to Cobaya names.
- ``latex_labels``: used for GetDist outputs.

Manual Access
-------------

.. code-block:: python

   info = bridge.info              # Cobaya info dict
   model = bridge.model            # Cobaya model
   loglike = model.loglike({'x': 0.0, 'y': 0.0})

Tips
----

- Keep priors bounded; GPry and Cobaya expect finite ranges for uniform/loguniform.
- Use GetDist (already in Cobaya) for plots and summaries: ``products = sampler.products(); samples = products["sample"]``.
- When using alternative names, ensure you still provide priors keyed by the original Discovery names.

See Also
--------

- :doc:`../api/gpry_interface` - API reference
- `Cobaya documentation <https://cobaya.readthedocs.io/>`_
- `GetDist documentation <https://getdist.readthedocs.io/>`_
