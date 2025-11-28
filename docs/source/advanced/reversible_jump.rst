Reversible-Jump MCMC
====================

Reversible-jump MCMC (RJ-MCMC) enables trans-dimensional sampling where the number of parameters can vary. This is useful for model selection and handling uncertainty in model complexity.

Overview
--------

**Reversible-Jump MCMC** extends standard MCMC to sample across models with different dimensionalities:

- **Fixed-dimension MCMC**: Samples from a single model with fixed parameters
- **Trans-dimensional MCMC**: Samples across multiple models with varying dimensions
- **Model selection**: Automatically determines which model best fits the data

Applications
------------

Common use cases:

1. **Number of signal sources**: How many gravitational wave sources are present in PTA data?
2. **Polynomial order**: What degree polynomial fits the data?
3. **Mixture components**: How many Gaussian components in a mixture?
4. **Change points**: How many regime changes in a time series?

Key Concepts
------------

Branches and Leaves
^^^^^^^^^^^^^^^^^^^

In the ``discoverysamplers`` RJMCMC interface:

- **Branch**: A type of model component (e.g., ``'cw'`` for continuous waves)
- **Leaf**: An instance of a component within a branch
- **nleaves_min/max**: Control the allowed range of component counts per branch

For example, with ``nleaves_min={'cw': 0}`` and ``nleaves_max={'cw': 5}``:

- Model :math:`M_0`: 0 CW sources
- Model :math:`M_1`: 1 CW source
- ...
- Model :math:`M_5`: 5 CW sources

Branch-Indexed Priors
^^^^^^^^^^^^^^^^^^^^^

Unlike standard priors, RJMCMC uses a nested dictionary structure:

.. code-block:: python

   # Standard priors (fixed-dimension)
   priors = {'x': ('uniform', -5, 5), 'y': ('uniform', -5, 5)}

   # RJMCMC priors (trans-dimensional)
   rj_priors = {
       'cw': {  # Branch name
           0: {},  # No parameters for 0 components
           1: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14)},
           2: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14)},
       }
   }

Basic Usage
-----------

Complete Example
^^^^^^^^^^^^^^^^

Here's a complete example using the ``RJ_Discovery_model`` and ``DiscoveryErynRJBridge``:

.. code-block:: python

   import numpy as np
   from discoverysamplers.eryn_RJ_interface import RJ_Discovery_model, DiscoveryErynRJBridge

   # 1. Define signal constructor (returns (delay_fn, param_names))
   def cw_signal_constructor(psr):
       """
       Signal constructor for continuous wave sources.

       Returns
       -------
       delay_fn : callable
           Function computing signal delay
       param_names : list of str
           Parameter names for this signal
       """
       def delay_fn(params):
           f = params['f']
           h = params['h']
           phi = params['phi']
           # Compute CW delay
           return compute_cw_delay(f, h, phi, psr.toas)

       return delay_fn, ['f', 'h', 'phi']

   # 2. Create RJ model wrapper
   rj_model = RJ_Discovery_model(
       signal_constructors=[cw_signal_constructor],  # One per branch
       pulsar=psr,
       variable_component_numbers=[0, 1, 2, 3, 4],   # Allowed component counts
   )

   # 3. Define branch-indexed priors
   priors = {
       'cw': {
           0: {},  # Empty for 0 components
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
           4: {
               'f': ('loguniform', 1e-9, 1e-7),
               'h': ('loguniform', 1e-20, 1e-14),
               'phi': ('uniform', 0, 2*np.pi),
           },
       }
   }

   # 4. Create the bridge
   bridge = DiscoveryErynRJBridge(
       discovery_model=rj_model,
       priors=priors,
       branch_names=['cw'],
       nleaves_min={'cw': 0},  # Minimum: 0 sources
       nleaves_max={'cw': 4},  # Maximum: 4 sources
   )

   # 5. Create sampler (parallel tempering recommended for RJMCMC)
   sampler = bridge.create_sampler(
       nwalkers=32,
       tempering_kwargs=dict(ntemps=4),
   )

   # 6. Initialize and run
   state = bridge.sample_priors(nwalkers=32, ntemps=4)
   sampler.run_mcmc(state, nsteps=20000, progress=True)

How RJ_Discovery_model Works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RJ_Discovery_model`` class:

1. **Pre-builds likelihoods** for each component count at initialization
2. **Caches likelihoods** to avoid recomputing during sampling
3. **Maps parameters** from Eryn's nested list format to Discovery dict format

.. code-block:: python

   # Eryn calls logL with nested lists:
   # params[branch_idx][component_idx] has shape (nwalkers, 1)

   # RJ_Discovery_model converts to dict format for Discovery:
   # {'f': value, 'h': value, 'phi': value}

Analyzing Results
-----------------

Component Count Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important result is the posterior distribution over component counts:

.. code-block:: python

   # Get number of leaves (components) per sample
   nleaves = sampler.get_nleaves()['cw']  # Shape: (nsteps, ntemps, nwalkers)

   # Extract cold chain (temp=0)
   cold_nleaves = nleaves[:, 0, :].flatten()

   # Plot histogram
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8, 5))
   counts = np.bincount(cold_nleaves.astype(int), minlength=5)
   probs = counts / counts.sum()

   ax.bar(range(len(probs)), probs)
   ax.set_xlabel('Number of CW sources')
   ax.set_ylabel('Posterior probability')
   ax.set_title('Model selection: number of sources')

   for i, p in enumerate(probs):
       ax.text(i, p + 0.02, f'{p:.2f}', ha='center')

   plt.tight_layout()
   plt.savefig('model_posterior.png')

Parameter Estimation
^^^^^^^^^^^^^^^^^^^^

Extract parameters for a specific number of components:

.. code-block:: python

   # Get chain for branch 'cw' from cold chain
   chain = sampler.get_chain()['cw'][:, 0, :, :, :]  # Cold chain only
   # Shape: (nsteps, nwalkers, max_leaves, ndim_per_leaf)

   # Get corresponding nleaves
   nleaves = sampler.get_nleaves()['cw'][:, 0, :]  # (nsteps, nwalkers)

   # Extract samples where exactly 2 sources were present
   n_target = 2
   mask = (nleaves == n_target)

   # Get parameters for these samples
   samples_2src = chain[mask][:, :n_target, :]  # (n_samples, 2, ndim_per_leaf)

   # Compute statistics
   print(f"Samples with {n_target} sources: {mask.sum()}")
   for i in range(n_target):
       print(f"Source {i}:")
       print(f"  f: {np.mean(samples_2src[:, i, 0]):.2e}")
       print(f"  h: {np.mean(samples_2src[:, i, 1]):.2e}")

Using the Corner Plot Helper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bridge provides a convenience method for corner plots:

.. code-block:: python

   # Make corner plot for samples with 2 components
   fig = bridge.make_corner_plot(
       sampler,
       n_components=2,
       discard=5000,  # Burn-in
   )
   fig.savefig('corner_2sources.png')

Best Practices
--------------

1. **Use parallel tempering**: RJMCMC benefits greatly from tempering to explore different dimensions

   .. code-block:: python

      sampler = bridge.create_sampler(
          nwalkers=32,
          tempering_kwargs=dict(ntemps=4),  # 4-8 temperatures typical
      )

2. **Start with reasonable nleaves_max**: Don't set too high; it increases computational cost

3. **Run long chains**: Trans-dimensional sampling needs more iterations than fixed-dimension

4. **Check acceptance rates**: Monitor birth/death move acceptance

5. **Validate on simulated data**: Test with known number of sources first

Troubleshooting
---------------

Sampler doesn't explore different dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Increase ``ntemps`` in parallel tempering
- Check that priors span reasonable ranges
- Verify likelihood returns finite values for all component counts

Poor mixing
^^^^^^^^^^^

- Run longer chains
- Increase number of walkers
- Adjust temperature ladder (``Tmax`` in ``tempering_kwargs``)

Memory issues
^^^^^^^^^^^^^

- Reduce ``nleaves_max``
- Reduce ``nwalkers`` or ``ntemps``
- Use checkpointing with ``eryn.backends.HDFBackend``

Example Notebooks
-----------------

See the ``examples/RJ_MCMC.ipynb`` notebook for:

- Complete toy example with Gaussian mixture
- Discovery PTA example with continuous waves
- Step-by-step analysis and plotting

See Also
--------

- :doc:`../user_guide/eryn_usage` - RJMCMC section in Eryn guide
- :doc:`../user_guide/prior_specification` - Branch-indexed prior format
- :doc:`../api/eryn_rj_interface` - API reference
- :doc:`parallel_tempering` - Parallel tempering details
- `Green (1995) <https://doi.org/10.1093/biomet/82.4.711>`_ - Original RJ-MCMC paper
