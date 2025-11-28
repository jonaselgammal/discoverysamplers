GPry Interface API
==================

.. automodule:: discoverysamplers.gpry_interface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

DiscoveryGPryCobayaBridge
--------------------------

.. autoclass:: discoverysamplers.gpry_interface.DiscoveryGPryCobayaBridge
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiscoveryGPryCobayaBridge.build_cobaya_info
      ~DiscoveryGPryCobayaBridge.get_cobaya_model
      ~DiscoveryGPryCobayaBridge.run_sampler

   .. rubric:: Attributes

   .. autosummary::

      ~DiscoveryGPryCobayaBridge.discovery_paramnames
      ~DiscoveryGPryCobayaBridge.sampled_names
      ~DiscoveryGPryCobayaBridge.fixed_names
      ~DiscoveryGPryCobayaBridge.n_sampled
      ~DiscoveryGPryCobayaBridge.n_fixed
      ~DiscoveryGPryCobayaBridge.sampled_prior_dict
      ~DiscoveryGPryCobayaBridge.fixed_param_dict
      ~DiscoveryGPryCobayaBridge.latex_labels
      ~DiscoveryGPryCobayaBridge.like_name

Helper Classes
--------------

ParsedPrior
^^^^^^^^^^^

.. autoclass:: discoverysamplers.gpry_interface.ParsedPrior
   :members:
   :undoc-members:

Method Details
--------------

build_cobaya_info
^^^^^^^^^^^^^^^^^

.. automethod:: discoverysamplers.gpry_interface.DiscoveryGPryCobayaBridge.build_cobaya_info

get_cobaya_model
^^^^^^^^^^^^^^^^

.. automethod:: discoverysamplers.gpry_interface.DiscoveryGPryCobayaBridge.get_cobaya_model

run_sampler
^^^^^^^^^^^

.. automethod:: discoverysamplers.gpry_interface.DiscoveryGPryCobayaBridge.run_sampler

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.gpry_interface import DiscoveryGPryCobayaBridge

   # Define model
   def model(params):
       return -0.5 * (params['x']**2 + params['y']**2)

   # Define priors
   priors = {
       'x': ('uniform', -5, 5),
       'y': ('uniform', -5, 5),
   }

   # Create bridge
   bridge = DiscoveryGPryCobayaBridge(
       discovery_model=model,
       priors=priors,
       like_name='gaussian_likelihood'
   )

   # Run sampler
   updated_info, sampler = bridge.run_sampler(
       max_samples=10000
   )

   # Access results
   products = sampler.products()
   samples = products["sample"]

Using Different Samplers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use MCMC instead of GPry
   updated_info, sampler = bridge.run_sampler(
       sampler='mcmc',
       max_samples=50000,
       Rminus1_stop=0.01
   )

   # Use PolyChord (if installed)
   updated_info, sampler = bridge.run_sampler(
       sampler='polychord',
       nlive=500
   )

Building Cobaya Info
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get Cobaya info dictionary
   info = bridge.build_cobaya_info()

   # Customize
   info['output'] = 'chains/my_run'
   info['sampler'] = {
       'mcmc': {
           'max_samples': 100000,
           'Rminus1_stop': 0.01,
       }
   }

   # Run manually
   from cobaya.run import run
   updated_info, sampler = run(info)

Using Cobaya Model
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get Cobaya model instance
   model = bridge.get_cobaya_model()

   # Test likelihood
   test_point = {'x': 1.0, 'y': 2.0}
   log_like = model.loglike(test_point)
   log_post = model.logpost(test_point)

   print(f"Log-likelihood: {log_like}")
   print(f"Log-posterior: {log_post}")

Parameter Remapping
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use alternative parameter names in Cobaya
   bridge = DiscoveryGPryCobayaBridge(
       discovery_model=model,
       priors={
           'mass1': ('uniform', 1, 3),
           'mass2': ('uniform', 1, 3),
       },
       alternative_paramnames={
           'mass1': 'm1',
           'mass2': 'm2',
       },
       latex_labels={
           'mass1': r'$m_1$',
           'mass2': r'$m_2$',
       }
   )

   # Cobaya uses 'm1', 'm2' but model receives 'mass1', 'mass2'
   updated_info, sampler = bridge.run_sampler(max_samples=10000)

Analyzing with GetDist
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from getdist import plots

   # Get samples
   products = sampler.products()
   samples = products["sample"]

   # Triangle plot
   g = plots.get_subplot_plotter()
   g.triangle_plot(samples, filled=True)
   g.export('corner.pdf')

   # Statistics
   for param in bridge.sampled_names:
       mean = samples.mean(param)
       std = samples.std(param)
       print(f"{param}: {mean:.3f} ± {std:.3f}")

See Also
--------

- :doc:`../user_guide/gpry_usage` - Usage guide
- `Cobaya documentation <https://cobaya.readthedocs.io/>`_ - Cobaya framework
