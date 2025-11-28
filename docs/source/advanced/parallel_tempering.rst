Parallel Tempering
==================

Parallel tempering (replica exchange MCMC) runs multiple chains at different temperatures to explore multimodal posteriors. Hot chains explore freely while cold chains sample the target posterior.

Basic Usage
-----------

.. code-block:: python

   from discoverysamplers.eryn_interface import DiscoveryErynBridge

   bridge = DiscoveryErynBridge(model, priors)

   # Create sampler with parallel tempering
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(
           ntemps=8,      # Number of temperature levels
           Tmax=20.0,     # Maximum temperature
       )
   )

   # Initialize for all temperatures
   initial_state = bridge.sample_priors(nwalkers=32, ntemps=8)

   # Run MCMC
   sampler.run_mcmc(initial_state, nsteps=10000, progress=True)

   # Extract cold chain samples (default)
   samples = sampler.get_chain(discard=1000, flat=True)

Temperature Configuration
-------------------------

.. code-block:: python

   # Geometric ladder (default)
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=5, Tmax=10.0)
   )

   # Adaptive ladder
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=8, Tmax=None, adaptive=True)
   )

   # Custom ladder
   import numpy as np
   custom_temps = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
   sampler = bridge.create_sampler(
       nwalkers=32,
       ntemps=len(custom_temps),
       betas=1.0 / custom_temps,
   )

**Guidelines**:

- Start with ``ntemps = 4-8``
- Rule of thumb: ``ntemps ≈ 2 * sqrt(ndim)``
- Typical ``Tmax = 10-50``

Diagnostics
-----------

.. code-block:: python

   # Check swap acceptance (target: 20-40%)
   swap_frac = sampler.tswap_acceptance_fraction
   for i, frac in enumerate(swap_frac):
       print(f"Swap {i} ↔ {i+1}: {frac:.2%}")

- **<10%**: Temperatures too far apart → increase ``ntemps`` or decrease ``Tmax``
- **>60%**: Temperatures too close → decrease ``ntemps`` or increase ``Tmax``

With Reversible-Jump
--------------------

.. code-block:: python

   from discoverysamplers.eryn_RJ_interface import DiscoveryErynRJBridge

   rj_bridge = DiscoveryErynRJBridge(
       discovery_model=rj_model,
       priors=rj_priors,
       branch_names=['cw'],
       nleaves_min={'cw': 0},
       nleaves_max={'cw': 5},
   )

   sampler = rj_bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=8, Tmax=30.0),
   )

See Also
--------

- :doc:`../user_guide/eryn_usage` - Basic Eryn usage
- :doc:`reversible_jump` - Trans-dimensional sampling
- `Eryn documentation <https://github.com/mikekatz04/Eryn>`_ - Eryn sampler
