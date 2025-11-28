Eryn Interface API
==================

.. automodule:: discoverysamplers.eryn_interface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

DiscoveryErynBridge
-------------------

.. autoclass:: discoverysamplers.eryn_interface.DiscoveryErynBridge
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiscoveryErynBridge.create_sampler
      ~DiscoveryErynBridge.sample_priors
      ~DiscoveryErynBridge.dict_to_array
      ~DiscoveryErynBridge.array_to_dict
      ~DiscoveryErynBridge.log_prob_fn

   .. rubric:: Attributes

   .. autosummary::

      ~DiscoveryErynBridge.discovery_paramnames
      ~DiscoveryErynBridge.sampled_names
      ~DiscoveryErynBridge.fixed_names
      ~DiscoveryErynBridge.n_sampled
      ~DiscoveryErynBridge.n_fixed
      ~DiscoveryErynBridge.ndim
      ~DiscoveryErynBridge.eryn_mapping
      ~DiscoveryErynBridge.latex_labels
      ~DiscoveryErynBridge.latex_list

Method Details
--------------

create_sampler
^^^^^^^^^^^^^^

.. automethod:: discoverysamplers.eryn_interface.DiscoveryErynBridge.create_sampler

sample_priors
^^^^^^^^^^^^^

.. automethod:: discoverysamplers.eryn_interface.DiscoveryErynBridge.sample_priors

dict_to_array
^^^^^^^^^^^^^

.. automethod:: discoverysamplers.eryn_interface.DiscoveryErynBridge.dict_to_array

array_to_dict
^^^^^^^^^^^^^

.. automethod:: discoverysamplers.eryn_interface.DiscoveryErynBridge.array_to_dict

log_prob_fn
^^^^^^^^^^^

.. automethod:: discoverysamplers.eryn_interface.DiscoveryErynBridge.log_prob_fn

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.eryn_interface import DiscoveryErynBridge

   # Define model
   def model(params):
       return -0.5 * (params['x']**2 + params['y']**2)

   # Define priors
   priors = {
       'x': ('uniform', -5, 5),
       'y': ('uniform', -5, 5),
   }

   # Create bridge
   bridge = DiscoveryErynBridge(model, priors)

   # Create sampler
   sampler = bridge.create_sampler(nwalkers=32)

   # Initialize and run
   initial = bridge.sample_priors(nwalkers=32)
   sampler.run_mcmc(initial, nsteps=10000)

With Parallel Tempering
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create sampler with parallel tempering
   sampler = bridge.create_sampler(
       nwalkers=32,
       ntemps=8,
       Tmax=20.0
   )

   # Initialize for all temperatures
   initial = bridge.sample_priors(nwalkers=32, ntemps=8)

   # Run sampling
   sampler.run_mcmc(initial, nsteps=10000)

   # Get samples from cold chain
   samples = sampler.get_chain(discard=1000, flat=True, temp=0)

Converting Between Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Dictionary to array
   param_dict = {'x': 1.0, 'y': 2.0}
   param_array = bridge.dict_to_array(param_dict)

   # Array to dictionary
   param_dict_back = bridge.array_to_dict(param_array)

See Also
--------

- :doc:`../user_guide/eryn_usage` - Usage guide
- :doc:`../advanced/parallel_tempering` - Advanced features
