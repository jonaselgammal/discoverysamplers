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
   from eryn.prior import uniform_dist
   from discoverysamplers.eryn_RJ_interface import RJ_Discovery_model, DiscoveryErynRJBridge

   # 1. Define signal constructor for variable components
   def cw_signal_constructor():
       """
       Signal constructor for continuous wave sources.
       
       Returns
       -------
       delay_fn : callable
           Function computing signal delay
       param_names : list of str
           Base parameter names for this signal
       """
       def delay_fn(params):
           # Compute CW delay based on params
           return compute_cw_delay(params)
       
       return delay_fn, ['log10_h0', 'log10_f0', 'ra', 'sindec']

   # 2. Create RJ model wrapper
   rj_model = RJ_Discovery_model(
       psrs=pulsars,
       fixed_components={'per_psr': {'base': make_fixed_components}},
       variable_components={'global': {'cw': (cw_signal_constructor, ['log10_h0', 'log10_f0', 'ra', 'sindec'])}},
       variable_component_numbers={'cw': (1, 4)},  # 1 to 4 sources
   )

   # 3. Define Eryn-format priors (branch -> param_index -> distribution)
   priors = {
       'cw': {
           0: uniform_dist(-20, -11),   # log10_h0
           1: uniform_dist(-9, -7),      # log10_f0
           2: uniform_dist(0, 2*np.pi),  # ra
           3: uniform_dist(-1, 1),       # sindec
       }
   }

   # 4. Create the bridge
   bridge = DiscoveryErynRJBridge(
       rj_model=rj_model,
       priors=priors,
   )

   # 5. Create sampler (parallel tempering recommended for RJMCMC)
   sampler = bridge.create_sampler(
       nwalkers=32,
       ntemps=4,
   )

   # 6. Initialize and run
   state = bridge.initialize_state(initial_nleaves=1)
   bridge.run_sampler(nsteps=20000, initial_state=state, progress=True)

How RJ_Discovery_model Works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RJ_Discovery_model`` class:

1. **Pre-builds likelihoods** for each component count at initialization
2. **Caches likelihoods** to avoid recomputing during sampling
3. **Maps parameters** from Eryn's nested list format to Discovery dict format

The model caches likelihoods for all configurations specified in ``variable_component_numbers``.

Analyzing Results
-----------------

Component Count Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important result is the posterior distribution over component counts:

.. code-block:: python

   # Get number of leaves (components) per sample
   nleaves = bridge.return_nleaves()  # Shape: (nsteps, ntemps, nwalkers)

   # Extract cold chain (temp=0)
   cold_nleaves = nleaves[:, 0, :].flatten()

   # Plot histogram
   bridge.plot_nleaves_histogram()

Parameter Estimation
^^^^^^^^^^^^^^^^^^^^

Extract parameters using the bridge methods:

.. code-block:: python

   # Get flattened samples (excludes inactive sources)
   flat_samples = bridge.return_flat_samples(temperature=0)

   # Get full chain with structure
   samples = bridge.return_sampled_samples(temperature=0)
   chain = samples['chain']  # (nsteps, nwalkers, nleaves_max, ndim)
   param_names = samples['names']

Using the Corner Plot Helper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bridge provides a convenience method for corner plots:

.. code-block:: python

   # Make corner plot of all active samples
   fig = bridge.plot_corner(temperature=0)
   fig.savefig('corner_rjmcmc.png')

Best Practices
--------------

1. **Use parallel tempering**: RJMCMC benefits greatly from tempering to explore different dimensions

   .. code-block:: python

      sampler = bridge.create_sampler(
          nwalkers=32,
          ntemps=4,  # 4-8 temperatures typical
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
