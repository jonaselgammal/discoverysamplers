Eryn Reversible-Jump Interface API
===================================

.. automodule:: discoverysamplers.eryn_RJ_interface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

RJ_Discovery_model
------------------

.. autoclass:: discoverysamplers.eryn_RJ_interface.RJ_Discovery_model
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

   The ``RJ_Discovery_model`` class wraps Discovery signal constructors for
   use with reversible-jump MCMC. It pre-builds and caches likelihoods for
   each possible number of components.

   .. rubric:: Key Features

   - **Automatic caching**: Builds likelihoods for all component counts at initialization
   - **Parameter handling**: Maps Eryn's ``(*params)`` format to Discovery dict format
   - **Signal constructors**: Works with Discovery signal constructors that return ``(delay_fn, param_names)``

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~RJ_Discovery_model.logL
      ~RJ_Discovery_model.get_likelihood

DiscoveryErynRJBridge
---------------------

.. autoclass:: discoverysamplers.eryn_RJ_interface.DiscoveryErynRJBridge
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   The ``DiscoveryErynRJBridge`` class bridges Discovery models to Eryn's
   reversible-jump MCMC sampler, enabling trans-dimensional sampling where
   the number of model components can vary.

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiscoveryErynRJBridge.create_sampler
      ~DiscoveryErynRJBridge.sample_priors
      ~DiscoveryErynRJBridge.make_corner_plot

   .. rubric:: Attributes

   .. autosummary::

      ~DiscoveryErynRJBridge.branch_names
      ~DiscoveryErynRJBridge.nleaves_min
      ~DiscoveryErynRJBridge.nleaves_max

Overview
--------

Reversible-jump MCMC allows the sampler to move between models with different
dimensionalities. This is useful for:

- **Model selection**: Automatically determine the number of signal components
- **Uncertainty in model complexity**: Account for uncertainty in the number of sources
- **Avoiding overfitting**: Let the data determine model complexity

Key Concepts
------------

Branch-Indexed Priors
^^^^^^^^^^^^^^^^^^^^^

Unlike standard sampling where priors map parameter names to distributions,
RJMCMC uses a branch-indexed structure:

.. code-block:: python

   # Standard prior format (fixed-dimensional)
   priors = {
       'x': ('uniform', -5.0, 5.0),
       'y': ('uniform', -5.0, 5.0),
   }

   # RJMCMC prior format (trans-dimensional)
   rj_priors = {
       'cw': {  # Branch name
           0: {},  # 0 components (empty)
           1: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14)},
           2: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14)},
       }
   }

Signal Constructors
^^^^^^^^^^^^^^^^^^^

Discovery signal constructors return ``(delay_function, param_names)`` tuples:

.. code-block:: python

   def my_signal_constructor(psr):
       def delay_fn(params):
           # Compute signal delay
           return delay

       return delay_fn, ['param1', 'param2']

The ``RJ_Discovery_model`` automatically handles building likelihoods
for each component count configuration.

Examples
--------

Complete RJMCMC Setup
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from discoverysamplers.eryn_RJ_interface import RJ_Discovery_model, DiscoveryErynRJBridge

   # 1. Define signal constructor (must return (delay_fn, param_names))
   def cw_signal(psr):
       def delay_fn(params):
           # Compute CW signal delay
           return compute_cw_delay(params['f'], params['h'], params['phi'], psr.toas)
       return delay_fn, ['f', 'h', 'phi']

   # 2. Create RJ model wrapper
   rj_model = RJ_Discovery_model(
       signal_constructors=[cw_signal],
       pulsar=psr,
       variable_component_numbers=[0, 1, 2, 3],
   )

   # 3. Define branch-indexed priors
   priors = {
       'cw': {
           0: {},
           1: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14), 'phi': ('uniform', 0, 2*np.pi)},
           2: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14), 'phi': ('uniform', 0, 2*np.pi)},
           3: {'f': ('loguniform', 1e-9, 1e-7), 'h': ('loguniform', 1e-20, 1e-14), 'phi': ('uniform', 0, 2*np.pi)},
       }
   }

   # 4. Create bridge
   bridge = DiscoveryErynRJBridge(
       discovery_model=rj_model,
       priors=priors,
       branch_names=['cw'],
       nleaves_min={'cw': 0},
       nleaves_max={'cw': 3},
   )

   # 5. Create sampler and run
   sampler = bridge.create_sampler(nwalkers=32, tempering_kwargs=dict(ntemps=4))
   state = bridge.sample_priors(nwalkers=32, ntemps=4)
   sampler.run_mcmc(state, nsteps=10000, progress=True)

   # 6. Analyze results
   nleaves = sampler.get_nleaves()['cw'][:, 0, :].flatten()
   print(f"Component count distribution: {np.bincount(nleaves.astype(int))}")

See Also
--------

- :doc:`../user_guide/eryn_usage` - User guide with RJMCMC section
- :doc:`../user_guide/prior_specification` - RJMCMC prior format
- :doc:`eryn_interface` - Standard (fixed-dimensional) Eryn interface
- :doc:`../examples/notebooks` - Example notebooks
