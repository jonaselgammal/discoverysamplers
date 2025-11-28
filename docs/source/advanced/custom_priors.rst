Custom Priors
=============

This guide covers advanced prior specifications beyond the standard distributions.

Overview
--------

While ``discoverysamplers`` provides built-in support for common priors (uniform, log-uniform, normal, fixed), you may need custom priors for:

- Non-standard distributions (e.g., Beta, Gamma, truncated distributions)
- Physical constraints (e.g., mass ratios, triangle inequalities)
- Informative priors from previous measurements
- Hierarchical priors

Callable Prior Interface
------------------------

Basic Callable Prior
^^^^^^^^^^^^^^^^^^^^

Create a custom prior by implementing a class with a ``logpdf`` method:

.. code-block:: python

   import numpy as np

   class CustomPrior:
       def __init__(self, param1, param2):
           """Initialize prior with hyperparameters."""
           self.param1 = param1
           self.param2 = param2
           # Optional: specify bounds for samplers that need them
           self.bounds = (lower, upper)

       def logpdf(self, value):
           """
           Compute log probability density.

           Parameters
           ----------
           value : float or array
               Parameter value(s)

           Returns
           -------
           float or array
               Log probability density
           """
           # Implement your prior here
           if self.bounds[0] <= value <= self.bounds[1]:
               return np.log(density_function(value))
           return -np.inf

   # Use with bridge
   priors = {
       'param1': CustomPrior(a=1.0, b=2.0),
       'param2': ('uniform', 0, 10),
   }

   bridge = DiscoveryNessaiBridge(model, priors)

Common Custom Priors
--------------------

Beta Distribution
^^^^^^^^^^^^^^^^^

Useful for parameters bounded in [0, 1]:

.. code-block:: python

   from scipy.stats import beta as beta_dist

   class BetaPrior:
       def __init__(self, a, b, loc=0, scale=1):
           """
           Beta distribution prior.

           Parameters
           ----------
           a, b : float
               Shape parameters (a, b > 0)
           loc : float
               Lower bound (default: 0)
           scale : float
               Range (default: 1)
           """
           self.dist = beta_dist(a, b, loc=loc, scale=scale)
           self.bounds = (loc, loc + scale)

       def logpdf(self, value):
           return self.dist.logpdf(value)

   # Example: Beta(2, 5) on [0, 1]
   priors = {
       'eccentricity': BetaPrior(a=2, b=5),
   }

Gamma Distribution
^^^^^^^^^^^^^^^^^^

For positive parameters:

.. code-block:: python

   from scipy.stats import gamma as gamma_dist

   class GammaPrior:
       def __init__(self, a, scale=1.0):
           """
           Gamma distribution prior.

           Parameters
           ----------
           a : float
               Shape parameter
           scale : float
               Scale parameter
           """
           self.dist = gamma_dist(a=a, scale=scale)
           self.bounds = (0, np.inf)

       def logpdf(self, value):
           if value < 0:
               return -np.inf
           return self.dist.logpdf(value)

   # Example: Gamma(2, scale=0.5)
   priors = {
       'rate_parameter': GammaPrior(a=2, scale=0.5),
   }

Truncated Gaussian
^^^^^^^^^^^^^^^^^^

Gaussian with hard bounds:

.. code-block:: python

   from scipy.stats import truncnorm

   class TruncatedNormalPrior:
       def __init__(self, mean, std, lower, upper):
           """
           Truncated normal distribution.

           Parameters
           ----------
           mean, std : float
               Mean and standard deviation of underlying Gaussian
           lower, upper : float
               Truncation bounds
           """
           # Convert to standard form
           a = (lower - mean) / std
           b = (upper - mean) / std
           self.dist = truncnorm(a, b, loc=mean, scale=std)
           self.bounds = (lower, upper)

       def logpdf(self, value):
           return self.dist.logpdf(value)

   # Example: N(0, 1) truncated to [-2, 2]
   priors = {
       'param': TruncatedNormalPrior(mean=0, std=1, lower=-2, upper=2),
   }

Mixture Prior
^^^^^^^^^^^^^

Mixture of distributions:

.. code-block:: python

   class MixturePrior:
       def __init__(self, components, weights):
           """
           Mixture of distributions.

           Parameters
           ----------
           components : list of distributions
               Each must have logpdf method
           weights : array
               Mixing weights (must sum to 1)
           """
           self.components = components
           self.weights = np.array(weights)
           assert np.allclose(self.weights.sum(), 1.0)

           # Bounds: union of component bounds
           self.bounds = (
               min(c.bounds[0] for c in components),
               max(c.bounds[1] for c in components)
           )

       def logpdf(self, value):
           # Log-sum-exp of weighted components
           log_probs = [
               np.log(w) + c.logpdf(value)
               for w, c in zip(self.weights, self.components)
           ]
           from scipy.special import logsumexp
           return logsumexp(log_probs)

   # Example: 50% U(0,1) + 50% U(9,10) (bimodal)
   from scipy.stats import uniform

   priors = {
       'param': MixturePrior(
           components=[
               uniform(loc=0, scale=1),
               uniform(loc=9, scale=1),
           ],
           weights=[0.5, 0.5]
       )
   }

JAX-Compatible Custom Priors
-----------------------------

For JAX-NS and JIT Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Custom priors must be JAX-compatible:

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit

   class JAXBetaPrior:
       def __init__(self, a, b):
           self.a = a
           self.b = b
           self.bounds = (0, 1)

       def logpdf(self, value):
           """JAX-compatible log-pdf."""
           # Beta distribution log-pdf
           from jax.scipy.special import betaln

           log_pdf = (
               (self.a - 1) * jnp.log(value) +
               (self.b - 1) * jnp.log(1 - value) -
               betaln(self.a, self.b)
           )

           # Use jnp.where instead of if/else
           return jnp.where(
               (value > 0) & (value < 1),
               log_pdf,
               -jnp.inf
           )

   # Use with JAX-NS
   from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge

   priors = {
       'param': JAXBetaPrior(a=2, b=5),
   }

   bridge = DiscoveryJAXNSBridge(jax_model, priors, jit=True)

Best Practices
--------------

1. **Always Specify Bounds**: Nested samplers need bounded priors

2. **Test Normalization**: Verify prior integrates to 1

3. **Use Log-Space**: Always work with log-probabilities for numerical stability

4. **JAX Compatibility**: Avoid Python control flow if using JAX

5. **Document Priors**: Clearly document prior choices and rationale

See Also
--------

- :doc:`../user_guide/prior_specification` - Standard priors
- :doc:`../user_guide/model_requirements` - Model requirements
- :doc:`performance` - Optimization tips
