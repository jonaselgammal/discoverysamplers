Nessai Interface API
====================

.. automodule:: discoverysamplers.nessai_interface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

DiscoveryNessaiBridge
---------------------

.. autoclass:: discoverysamplers.nessai_interface.DiscoveryNessaiBridge
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiscoveryNessaiBridge.run_sampler
      ~DiscoveryNessaiBridge.dict_to_livepoint
      ~DiscoveryNessaiBridge.livepoint_to_dict
      ~DiscoveryNessaiBridge.sample_priors

   .. rubric:: Attributes

   .. autosummary::

      ~DiscoveryNessaiBridge.discovery_paramnames
      ~DiscoveryNessaiBridge.sampled_names
      ~DiscoveryNessaiBridge.fixed_names
      ~DiscoveryNessaiBridge.n_sampled
      ~DiscoveryNessaiBridge.n_fixed
      ~DiscoveryNessaiBridge.sampled_prior_dict
      ~DiscoveryNessaiBridge.fixed_param_dict
      ~DiscoveryNessaiBridge.latex_labels

Helper Classes
--------------

ParsedPrior
^^^^^^^^^^^

.. autoclass:: discoverysamplers.nessai_interface.ParsedPrior
   :members:
   :undoc-members:

DiscoveryNessaiModel
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: discoverysamplers.nessai_interface.DiscoveryNessaiModel
   :members:
   :undoc-members:
   :show-inheritance:

Method Details
--------------

run_sampler
^^^^^^^^^^^

.. automethod:: discoverysamplers.nessai_interface.DiscoveryNessaiBridge.run_sampler

dict_to_livepoint
^^^^^^^^^^^^^^^^^

.. automethod:: discoverysamplers.nessai_interface.DiscoveryNessaiBridge.dict_to_livepoint

livepoint_to_dict
^^^^^^^^^^^^^^^^^

.. automethod:: discoverysamplers.nessai_interface.DiscoveryNessaiBridge.livepoint_to_dict

sample_priors
^^^^^^^^^^^^^

.. automethod:: discoverysamplers.nessai_interface.DiscoveryNessaiBridge.sample_priors

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

   # Define model
   def model(params):
       return -0.5 * (params['x']**2 + params['y']**2)

   # Define priors
   priors = {
       'x': ('uniform', -5, 5),
       'y': ('uniform', -5, 5),
   }

   # Create bridge with JIT
   bridge = DiscoveryNessaiBridge(
       discovery_model=model,
       priors=priors,
       jit=True
   )

   # Run sampler
   results = bridge.run_sampler(
       nlive=1000,
       output='output/nessai/',
       resume=False
   )

   # Access results
   print(f"Log evidence: {results['logZ']}")

With Flow Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Custom flow configuration
   results = bridge.run_sampler(
       nlive=1000,
       output='output/custom_flow/',
       flow_config={
           'model_config': {
               'n_blocks': 8,
               'n_neurons': 64,
           }
       }
   )

Resuming Runs
^^^^^^^^^^^^^

.. code-block:: python

   # Initial run
   results = bridge.run_sampler(
       nlive=1000,
       output='output/checkpoint/',
       resume=False,
       max_iteration=5000
   )

   # Resume
   results = bridge.run_sampler(
       nlive=1000,
       output='output/checkpoint/',
       resume=True,
       max_iteration=10000
   )

Working with Samples
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Convert dictionary to structured array (live point)
   param_dict = {'x': 1.0, 'y': 2.0}
   livepoint = bridge.dict_to_livepoint(param_dict)

   # Convert back
   param_dict_back = bridge.livepoint_to_dict(livepoint)

   # Sample from priors
   prior_samples = bridge.sample_priors(n=1000)

See Also
--------

- :doc:`../user_guide/nessai_usage` - Usage guide
- :doc:`../advanced/performance` - Performance optimization
