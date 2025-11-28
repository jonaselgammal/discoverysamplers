Model Requirements
==================

This page describes the requirements for Discovery models to work with ``discoverysamplers`` bridge interfaces.

Basic Requirements
------------------

Your model must satisfy these fundamental requirements:

1. **Callable**: The model must be a callable object (function, method, or class with ``__call__``)
2. **Dictionary Input**: Must accept a dictionary mapping parameter names (strings) to parameter values
3. **Scalar Output**: Must return a single scalar value representing the log-likelihood
4. **Deterministic**: Should return the same output for the same input (for reproducibility)

**Note**: For Discovery likelihoods, you can pass either the ``likelihood`` object or ``likelihood.logL`` — all bridges automatically extract the callable if you pass the object.

Valid Model Signatures
----------------------

Discovery Likelihood (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to use ``discoverysamplers`` is with Discovery likelihoods:

.. code-block:: python

   import discovery as ds

   # Load pulsar and create likelihood
   psr = ds.Pulsar.read_feather('path/to/pulsar.feather')
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),
       ds.makegp_timing(psr, svd=True),
   ])

   # You can pass either likelihood or likelihood.logL to bridges
   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

   bridge = DiscoveryNessaiBridge(
       discovery_model=likelihood,  # Bridge extracts .logL automatically
       priors=priors
   )
   # OR equivalently:
   bridge = DiscoveryNessaiBridge(
       discovery_model=likelihood.logL,
       priors=priors
   )

Simple Function
^^^^^^^^^^^^^^^

The most straightforward model is a simple function:

.. code-block:: python

   def my_model(params: dict) -> float:
       """
       Compute log-likelihood from parameters.

       Parameters
       ----------
       params : dict
           Dictionary mapping parameter names to values

       Returns
       -------
       float
           Log-likelihood value
       """
       x = params['x']
       y = params['y']

       return -0.5 * (x**2 + y**2)

Function with Additional Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your model needs additional data or configuration, you can use closures or partial functions:

.. code-block:: python

   import numpy as np
   from functools import partial

   def model_with_data(params: dict, data: np.ndarray, noise_std: float) -> float:
       """Model that requires data."""
       signal = params['amplitude'] * np.sin(2 * np.pi * params['frequency'] * data)
       residuals = signal - data
       return -0.5 * np.sum((residuals / noise_std)**2)

   # Create a partial function with fixed data
   data = np.random.randn(100)
   model = partial(model_with_data, data=data, noise_std=0.1)

   # Now model(params) has the correct signature
   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
   bridge = DiscoveryNessaiBridge(model, priors)

Callable Class
^^^^^^^^^^^^^^

For complex models, use a class with a ``__call__`` method:

.. code-block:: python

   class MyComplexModel:
       def __init__(self, data, config):
           """Initialize model with data and configuration."""
           self.data = data
           self.config = config
           self._precompute()

       def _precompute(self):
           """Pre-compute expensive quantities."""
           self.data_fft = np.fft.fft(self.data)
           # ... other precomputations

       def __call__(self, params: dict) -> float:
           """Compute log-likelihood."""
           # Use precomputed quantities for efficiency
           signal = self._generate_signal(params)
           return self._compute_log_likelihood(signal)

       def _generate_signal(self, params):
           # Implementation details
           pass

       def _compute_log_likelihood(self, signal):
           # Implementation details
           pass

   # Use the class instance as the model
   model = MyComplexModel(data, config)
   bridge = DiscoveryNessaiBridge(model, priors)

Parameter Dictionary Structure
-------------------------------

Input Dictionary
^^^^^^^^^^^^^^^^

The input ``params`` dictionary contains all sampled and fixed parameters:

.. code-block:: python

   params = {
       'mass': 1.5,           # float
       'distance': 100.0,     # float
       'sky_loc': [0.5, 1.2], # Can be array-like for vector parameters
   }

**Key Points**:

- Keys are parameter name strings
- Values are typically floats or NumPy arrays
- All sampled parameters (from ``priors``) will be present
- Fixed parameters are automatically injected by the bridge
- Parameter order doesn't matter (dictionary is unordered)

Handling Vector Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your model has vector-valued parameters:

.. code-block:: python

   def model_with_vectors(params):
       """Model with vector parameters."""
       # Scalar parameters
       mass = params['mass']

       # Vector parameters (arrays)
       sky_location = params['sky_location']  # Shape: (2,)
       ra, dec = sky_location

       # 2D parameters
       covariance = params['covariance']  # Shape: (3, 3)

       return compute_log_likelihood(mass, ra, dec, covariance)

   # Priors for vector parameters
   priors = {
       'mass': ('uniform', 1.0, 3.0),
       'sky_location': [
           ('uniform', 0, 2*np.pi),    # RA
           ('uniform', -np.pi/2, np.pi/2),  # Dec
       ],
       # For complex structures, use fixed parameters
       'covariance': ('fixed', np.eye(3)),
   }

Return Value
------------

Log-Likelihood
^^^^^^^^^^^^^^

The model must return the **log-likelihood** (not likelihood or chi-squared):

.. code-block:: python

   def correct_model(params):
       # Compute chi-squared or similar
       chi2 = compute_chi_squared(params)

       # Return log-likelihood
       return -0.5 * chi2

   def incorrect_model(params):
       # DON'T return likelihood
       likelihood = np.exp(-0.5 * chi2)  # Wrong!
       return likelihood

       # DON'T return chi-squared
       return chi2  # Wrong!

Special Values
^^^^^^^^^^^^^^

Use ``-np.inf`` to indicate impossible parameter combinations:

.. code-block:: python

   import numpy as np

   def model_with_constraints(params):
       mass1 = params['mass1']
       mass2 = params['mass2']

       # Physical constraint: mass1 >= mass2
       if mass1 < mass2:
           return -np.inf

       # Numerical stability: avoid log(0)
       if mass1 < 1e-10:
           return -np.inf

       # Otherwise compute likelihood
       return compute_log_likelihood(mass1, mass2)

**Never** return ``+np.inf`` or ``np.nan`` as these will break the samplers.

Performance Considerations
--------------------------

JAX Compatibility
^^^^^^^^^^^^^^^^^

For best performance with JAX-based samplers (JAX-NS, or when using ``jit=True``):

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit

   def jax_model(params):
       """JAX-compatible model using jax.numpy."""
       x = params['x']
       y = params['y']

       # Use jax.numpy instead of numpy
       return -0.5 * (jnp.square(x) + jnp.square(y))

   # Optionally JIT-compile the model
   jax_model_compiled = jit(jax_model)

   bridge = DiscoveryJAXNSBridge(jax_model_compiled, priors, jit=True)

**JAX Tips**:

- Use ``jax.numpy`` instead of ``numpy``
- Avoid Python control flow (if/else); use ``jnp.where`` instead
- Avoid in-place array updates
- See `JAX documentation <https://jax.readthedocs.io/>`_ for more details

Vectorization
^^^^^^^^^^^^^

Some samplers (especially JAX-NS) can evaluate the likelihood for multiple parameter sets simultaneously:

.. code-block:: python

   def vectorized_model(params):
       """Model that supports vectorized evaluation."""
       x = params['x']  # Could be shape (N,) instead of scalar
       y = params['y']  # Could be shape (N,)

       # Operations work element-wise
       return -0.5 * (x**2 + y**2)  # Returns shape (N,)

   bridge = DiscoveryJAXNSBridge(vectorized_model, priors)
   bridge.configure_array_api(order=['x', 'y'])  # Enable vectorization

Caching and Memoization
^^^^^^^^^^^^^^^^^^^^^^^

If your likelihood involves expensive computations, consider caching:

.. code-block:: python

   from functools import lru_cache

   class CachedModel:
       def __init__(self):
           # Cache expensive precomputations
           self._cache = {}

       def __call__(self, params):
           # Create a hashable key from params
           key = tuple(sorted(params.items()))

           if key not in self._cache:
               self._cache[key] = self._compute_expensive(params)

           return self._cache[key]

       def _compute_expensive(self, params):
           # Expensive likelihood computation
           return log_likelihood

   model = CachedModel()

**Warning**: Ensure cache keys are truly unique to avoid incorrect results.

Common Patterns
---------------

Pattern 1: Separating Model Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class ModularModel:
       def __init__(self, data):
           self.data = data

       def __call__(self, params):
           signal = self.signal_model(params)
           noise = self.noise_model(params)
           return self.log_likelihood(signal, noise)

       def signal_model(self, params):
           """Generate signal from parameters."""
           return params['amplitude'] * np.sin(2 * np.pi * params['frequency'] * self.data)

       def noise_model(self, params):
           """Model noise properties."""
           return params['noise_level']

       def log_likelihood(self, signal, noise):
           """Compute log-likelihood from signal and noise."""
           residuals = signal - self.data
           return -0.5 * np.sum((residuals / noise)**2)

Pattern 2: Model with Units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from astropy import units as u

   class PhysicalModel:
       def __init__(self, data, units_dict):
           """
           Parameters
           ----------
           units_dict : dict
               Maps parameter names to astropy units
           """
           self.data = data
           self.units = units_dict

       def __call__(self, params):
           # Attach units
           mass = params['mass'] * self.units['mass']
           distance = params['distance'] * self.units['distance']

           # Compute physical quantities with unit checking
           quantity = (mass / distance**2).to(u.solMass / u.Mpc**2)

           # Return dimensionless log-likelihood
           return float(-0.5 * quantity.value**2)

Pattern 3: Hierarchical Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def hierarchical_model(params):
       """Model with hierarchical parameters."""
       # Population-level parameters
       mu = params['population_mean']
       sigma = params['population_std']

       # Individual-level parameters
       individuals = [params[f'individual_{i}'] for i in range(10)]

       # Population prior
       log_prior = -0.5 * np.sum([(ind - mu)**2 / sigma**2 for ind in individuals])

       # Individual likelihoods
       log_like = sum([individual_likelihood(ind) for ind in individuals])

       return log_prior + log_like

Error Handling
--------------

Graceful Failure
^^^^^^^^^^^^^^^^

Handle potential errors gracefully:

.. code-block:: python

   def robust_model(params):
       try:
           result = compute_likelihood(params)

           # Check for invalid results
           if not np.isfinite(result):
               return -np.inf

           return result

       except Exception as e:
           # Log the error for debugging
           print(f"Error in likelihood: {e}, params: {params}")
           return -np.inf

Numerical Stability
^^^^^^^^^^^^^^^^^^^

Avoid numerical issues:

.. code-block:: python

   def stable_model(params):
       x = params['x']

       # Avoid division by zero
       if abs(x) < 1e-10:
           return -np.inf

       # Use log-space for products
       # Instead of: p = p1 * p2 * p3
       # Use: log_p = log_p1 + log_p2 + log_p3
       log_p = np.log(p1) + np.log(p2) + np.log(p3)

       # Use logsumexp for sums in log-space
       from scipy.special import logsumexp
       log_sum = logsumexp([log_a, log_b, log_c])

       return log_p + log_sum

Testing Your Model
------------------

Before running samplers, test your model:

.. code-block:: python

   def test_model():
       """Test model with sample parameters."""
       # Test with typical values
       params = {'x': 1.0, 'y': 2.0}
       result = my_model(params)

       assert np.isfinite(result), "Model returned non-finite value"
       assert isinstance(result, (float, np.floating)), "Model must return scalar"

       # Test boundary cases
       edge_params = {'x': -5.0, 'y': 5.0}  # Prior bounds
       result = my_model(edge_params)
       assert np.isfinite(result), "Model failed at prior boundaries"

       # Test multiple calls (reproducibility)
       result1 = my_model(params)
       result2 = my_model(params)
       assert result1 == result2, "Model is not deterministic"

       print("Model tests passed!")

   test_model()

RJMCMC Model Requirements
-------------------------

For reversible jump MCMC (trans-dimensional sampling), models have different requirements:

**Signal Constructor Pattern**:

RJMCMC models typically use signal constructors that return a ``(delay_function, param_names)`` tuple:

.. code-block:: python

   def my_signal_constructor(psr):
       """
       Signal constructor for RJMCMC.

       Parameters
       ----------
       psr : Pulsar
           Pulsar object

       Returns
       -------
       delay_fn : callable
           Function that computes signal delay given parameters
       param_names : list of str
           Names of parameters for this signal component
       """
       def delay_fn(params):
           # Compute signal delay from parameters
           f = params['f']
           h = params['h']
           phi = params['phi']
           return compute_cw_delay(f, h, phi, psr.toas)

       return delay_fn, ['f', 'h', 'phi']

**Using RJ_Discovery_model**:

The ``RJ_Discovery_model`` wrapper handles caching likelihoods for different component counts:

.. code-block:: python

   from discoverysamplers.eryn_RJ_interface import RJ_Discovery_model

   rj_model = RJ_Discovery_model(
       signal_constructors=[my_signal_constructor],
       pulsar=psr,
       variable_component_numbers=[0, 1, 2, 3],  # Allowed component counts
   )

   # The model caches likelihoods for each configuration
   # logL receives parameters as nested lists:
   # params[branch_idx][component_idx] is shape (nwalkers, 1)

**logL Signature for RJMCMC**:

.. code-block:: python

   class RJ_Discovery_model:
       def logL(self, *params):
           """
           Compute log-likelihood for variable number of components.

           Parameters
           ----------
           *params : tuple of lists
               params[i] is a list of arrays for branch i
               params[i][j] has shape (nwalkers, 1) for component j

           Returns
           -------
           array
               Log-likelihood values, shape (nwalkers,)
           """
           # Determine active components from params structure
           cw_params = params[0]  # First branch
           n_sources = len(cw_params)

           # Select pre-cached likelihood for this configuration
           return self.likelihoods[n_sources](combined_params)

See :doc:`eryn_usage` for complete RJMCMC examples.

Checklist
---------

Before using your model with ``discoverysamplers``, verify:

- [ ] Model accepts a dictionary and returns a scalar
- [ ] Model returns log-likelihood (not likelihood or chi-squared)
- [ ] Model returns ``-np.inf`` for invalid parameters (not exceptions)
- [ ] Model is deterministic (same input → same output)
- [ ] Model works with all parameters in the prior dictionary
- [ ] Model handles fixed parameters correctly (they're auto-injected)
- [ ] Model is numerically stable across the prior range
- [ ] If using JAX: model uses ``jax.numpy`` and avoids unsupported operations
- [ ] Model has been tested with sample parameter values

See Also
--------

- :doc:`prior_specification` - How to specify priors for parameters (including RJMCMC)
- :doc:`quickstart` - Basic usage examples
- :doc:`eryn_usage` - RJMCMC and parallel tempering
- :doc:`../advanced/performance` - Performance optimization guide
