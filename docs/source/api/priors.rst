Priors Module API
==================

.. automodule:: discoverysamplers.priors
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``priors`` module provides common functionality for parsing prior specifications
and adapting Discovery models to sampler interfaces. This module contains utilities
shared across all sampler bridges.

Classes
-------

ParsedPrior
^^^^^^^^^^^

.. autoclass:: discoverysamplers.priors.ParsedPrior
   :members:
   :undoc-members:

   Data class representing a parsed prior distribution with unified format.

PriorParsingError
^^^^^^^^^^^^^^^^^

.. autoclass:: discoverysamplers.priors.PriorParsingError
   :members:
   :undoc-members:

   Exception raised when a prior specification cannot be parsed.

Functions
---------

_parse_single_prior
^^^^^^^^^^^^^^^^^^^

.. autofunction:: discoverysamplers.priors._parse_single_prior

_split_priors
^^^^^^^^^^^^^

.. autofunction:: discoverysamplers.priors._split_priors

Internal Classes
----------------

_DiscoveryAdapter
^^^^^^^^^^^^^^^^^

.. autoclass:: discoverysamplers.priors._DiscoveryAdapter
   :members:
   :undoc-members:
   :show-inheritance:

   Internal adapter class for wrapping Discovery models with JIT compilation
   and optional vectorization support.

Type Definitions
----------------

.. py:data:: ParamName
   :type: str

   Type alias for parameter names.

.. py:data:: PriorSpec
   :type: Union[Mapping[str, Any], Tuple[str, float, float], Tuple[str, float, float, float], Tuple[str, float], Callable, float]

   Type alias for prior specifications. Can be:

   - Dict: ``{'dist': 'uniform', 'min': 0, 'max': 1}``
   - Tuple: ``('uniform', 0, 1)``, ``('loguniform', 1e-3, 1)``, ``('normal', 0, 1)``, ``('fixed', 0.5)``
   - Callable: Object with ``logpdf(value)`` method
   - Float: Interpreted as fixed value

Examples
--------

Parsing a Single Prior
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.priors import _parse_single_prior

   # Parse uniform prior from tuple
   parsed = _parse_single_prior('mass', ('uniform', 1.0, 3.0))
   print(parsed.dist_type)  # 'uniform'
   print(parsed.bounds)     # (1.0, 3.0)

   # Parse from dict
   parsed = _parse_single_prior('freq', {'dist': 'loguniform', 'min': 1, 'max': 1000})
   print(parsed.dist_type)  # 'loguniform'
   print(parsed.bounds)     # (1, 1000)

   # Fixed parameter
   parsed = _parse_single_prior('z', ('fixed', 0.2))
   print(parsed.dist_type)  # 'fixed'
   print(parsed.value)      # 0.2

Splitting Priors
^^^^^^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.priors import _split_priors

   priors = {
       'mass1': ('uniform', 1.0, 3.0),
       'mass2': ('uniform', 1.0, 3.0),
       'distance': ('loguniform', 10, 1000),
       'z': ('fixed', 0.2),
   }

   sampled_names, fixed_params, bounds, logprior_fns = _split_priors(priors)

   print(sampled_names)    # ['mass1', 'mass2', 'distance']
   print(fixed_params)     # {'z': 0.2}
   print(bounds['mass1'])  # (1.0, 3.0)

   # Evaluate log-prior
   log_p = logprior_fns['mass1'](2.0)  # Returns -log(3.0-1.0) = -0.693...

Using Discovery Adapter
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.priors import _DiscoveryAdapter
   import discovery as ds

   # Create Discovery likelihood
   psr = ds.Pulsar.read_feather('pulsar.feather')
   likelihood = ds.PulsarLikelihood([...])

   # Wrap with adapter
   fixed_params = {'param1': 1.0}
   adapter = _DiscoveryAdapter(
       model=likelihood.logL,
       fixed_params=fixed_params,
       jit=True
   )

   # Evaluate likelihood (fixed params auto-injected)
   params = {'param2': 2.0, 'param3': 3.0}
   log_L = adapter.log_likelihood(params)

See Also
--------

- :doc:`../user_guide/prior_specification` - User guide on prior specifications
- :doc:`../advanced/custom_priors` - Creating custom prior distributions
