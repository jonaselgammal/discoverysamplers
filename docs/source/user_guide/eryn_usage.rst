Eryn MCMC Sampler
=================

Interface to `Eryn <https://github.com/mikekatz04/Eryn>`_, an ensemble MCMC sampler with optional parallel tempering and reversible jump (RJMCMC) capabilities. Use it for posterior sampling when you do not need evidence estimates.

Minimal Run
-----------

.. code-block:: python

   import numpy as np
   from discoverysamplers.eryn_interface import DiscoveryErynBridge

   def my_model(params):
       x, y = params['x'], params['y']
       return -0.5 * (x**2 + y**2)

   priors = {'x': ('uniform', -5.0, 5.0), 'y': ('uniform', -5.0, 5.0)}

   bridge = DiscoveryErynBridge(model=my_model, priors=priors)
   sampler = bridge.create_sampler(nwalkers=32)

   p0 = bridge.sample_priors(nwalkers=32)  # initialize from prior
   sampler.run_mcmc(p0, nsteps=5000, progress=True)

   samples = sampler.get_chain(discard=1000, flat=True)
   x_samples = samples[:, bridge.eryn_mapping['x']]
   print(f"x mean = {np.mean(x_samples):.3f}")

Parallel Tempering
------------------

Parallel tempering helps sample multi-modal posteriors by running chains at different "temperatures." Hot chains explore more broadly while cold chains sample the target distribution.

**Basic Usage**:

.. code-block:: python

   # Enable parallel tempering with tempering_kwargs
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=8)  # 8 temperature chains
   )

   # Initial state must have shape (ntemps, nwalkers, ndim)
   p0 = bridge.sample_priors(nwalkers=32, ntemps=8)
   sampler.run_mcmc(p0, nsteps=5000, progress=True)

   # get_chain returns only the cold chain (temp=0) by default
   samples = sampler.get_chain(discard=1000, flat=True)

**Configuration Options**:

.. code-block:: python

   # Fine-tune tempering behavior
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(
           ntemps=8,         # Number of temperature levels
           Tmax=None,        # Maximum temperature (None = adaptive)
           adaptive=True,    # Adapt temperatures during run
       )
   )

**Tips for Parallel Tempering**:

- Start with ``ntemps=4-8`` and increase if chains don't mix well
- Target swap acceptance rates around 20-40%
- Use more temperatures for highly multi-modal posteriors
- Monitor ``sampler.acceptance_fraction`` per temperature

Reversible Jump MCMC (RJMCMC)
-----------------------------

RJMCMC enables trans-dimensional sampling where the number of model components can vary. This is useful for model selection problems like counting the number of gravitational wave sources in PTA data.

**Setting Up RJMCMC**:

.. code-block:: python

   import numpy as np
   from discoverysamplers.eryn_RJ_interface import RJ_Discovery_model, DiscoveryErynRJBridge

   # 1. Create your base Discovery model (must return (delay_fn, param_names) when called)
   # This is typically a signal constructor like cw_delay
   def signal_constructor(psr):
       # Returns (delay_function, param_names)
       return delay_fn, ['f', 'h', 'phi']

   # 2. Wrap in RJ_Discovery_model
   rj_model = RJ_Discovery_model(
       signal_constructors=[signal_constructor],  # One per branch
       pulsar=psr,
       variable_component_numbers=[0, 1, 2, 3],   # Allowed component counts
   )

   # 3. Define branch-indexed priors
   priors = {
       'cw': {
           0: {},  # No parameters when 0 components
           1: {
               'f': ('loguniform', 1e-9, 1e-7),
               'h': ('loguniform', 1e-20, 1e-14),
               'phi': ('uniform', 0, 2*np.pi),
           },
           2: {
               'f': ('loguniform', 1e-9, 1e-7),
               'h': ('loguniform', 1e-20, 1e-14),
               'phi': ('uniform', 0, 2*np.pi),
           },
           3: {
               'f': ('loguniform', 1e-9, 1e-7),
               'h': ('loguniform', 1e-20, 1e-14),
               'phi': ('uniform', 0, 2*np.pi),
           },
       }
   }

   # 4. Create bridge
   bridge = DiscoveryErynRJBridge(
       discovery_model=rj_model,
       priors=priors,
       branch_names=['cw'],
       nleaves_min={'cw': 0},  # Minimum components
       nleaves_max={'cw': 3},  # Maximum components
   )

   # 5. Create sampler with RJMCMC moves enabled
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=4),  # Often want tempering for RJMCMC
   )

**Running and Analyzing RJMCMC**:

.. code-block:: python

   # Initialize from priors
   state = bridge.sample_priors(nwalkers=32, ntemps=4)
   sampler.run_mcmc(state, nsteps=10000, progress=True)

   # Get number of components per sample
   nleaves = sampler.get_nleaves()['cw']  # Shape: (nsteps, ntemps, nwalkers)

   # Plot component count histogram
   import matplotlib.pyplot as plt
   cold_nleaves = nleaves[:, 0, :].flatten()  # Cold chain only
   plt.hist(cold_nleaves, bins=np.arange(-0.5, 4.5), density=True)
   plt.xlabel('Number of components')
   plt.ylabel('Posterior probability')

**Key RJMCMC Concepts**:

- **Branches**: Groups of parameters that can appear multiple times (e.g., ``'cw'`` for continuous waves)
- **Leaves**: Individual instances within a branch (e.g., each CW source is a leaf)
- **``nleaves_min/max``**: Control the allowed range of component counts
- **Model format**: Your ``logL(*params)`` receives nested lists where ``params[i][j]`` contains parameters for component ``j`` of branch ``i``

See the ``examples/RJ_MCMC.ipynb`` notebook for a complete working example.

Key Options
-----------

**For ``create_sampler()``**:

- ``nwalkers``: at least ``2 * ndim`` (typ. 32–64).
- ``tempering_kwargs``: dict with ``ntemps``, ``Tmax``, ``adaptive`` for parallel tempering.
- ``moves``: pass Eryn move objects/weights to customize proposals.
- ``backend``: use ``eryn.backends.HDFBackend`` to checkpoint chains.

**For RJMCMC** (``DiscoveryErynRJBridge``):

- ``branch_names``: list of branch names (e.g., ``['cw']``).
- ``nleaves_min/max``: dicts mapping branch names to min/max component counts.
- RJMCMC moves are automatically enabled.

Quick Diagnostics
-----------------

.. code-block:: python

   print(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
   # Autocorr may fail for short runs; guard with try/except
   try:
       tau = sampler.get_autocorr_time()
       print(f"Autocorr time: {tau}")
   except Exception:
       pass

Tips
----

- Provide priors for all Discovery parameters; missing entries raise errors.
- Use parallel tempering when you suspect multimodality; target swap rates ~20–40%.
- Start from priors for broad exploration; from a Gaussian ball if you have a good initial guess.

See Also
--------

- :doc:`prior_specification` - Prior format for RJMCMC
- :doc:`../api/eryn_interface` - API reference for fixed-dimensional MCMC
- :doc:`../api/eryn_RJ_interface` - API reference for RJMCMC
- :doc:`../examples/notebooks` - Example notebooks including RJMCMC
- `Eryn documentation <https://github.com/mikekatz04/Eryn>`_
