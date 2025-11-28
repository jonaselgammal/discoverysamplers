Extending discoverysamplers
===========================

This guide shows how to extend ``discoverysamplers`` to support new samplers or customize existing interfaces.

Creating a New Sampler Interface
---------------------------------

Basic Structure
^^^^^^^^^^^^^^^

All sampler interfaces follow a common pattern:

.. code-block:: python

   from typing import Dict, Any, Optional
   import numpy as np

   class DiscoveryNewSamplerBridge:
       """Bridge interface for NewSampler."""

       def __init__(
           self,
           discovery_model,
           priors: Dict[str, Any],
           latex_labels: Optional[Dict[str, str]] = None,
           jit: bool = False
       ):
           """
           Initialize bridge.

           Parameters
           ----------
           discovery_model : callable
               Model function accepting parameter dict, returning log-likelihood
           priors : dict
               Prior specifications
           latex_labels : dict, optional
               LaTeX labels for parameters
           jit : bool
               Enable JIT compilation if applicable
           """
           self.model = discovery_model
           self.jit = jit

           # Parse priors (separate fixed vs sampled)
           self._parse_priors(priors)

           # Setup parameter names and mappings
           self._setup_parameters()

           # Store LaTeX labels
           self.latex_labels = latex_labels or {}

       def _parse_priors(self, priors):
           """Parse prior specifications."""
           # Implement prior parsing
           # Separate fixed parameters from sampled parameters
           pass

       def _setup_parameters(self):
           """Setup parameter name lists and mappings."""
           # Create parameter name lists
           # Setup index mappings
           pass

       def run_sampler(self, **kwargs):
           """
           Run the sampler.

           Parameters
           ----------
           **kwargs
               Sampler-specific options

           Returns
           -------
           dict
               Results dictionary
           """
           # Implement sampler execution
           pass

Example: Minimal Sampler Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from discoverysamplers.nessai_interface import _split_priors, _parse_single_prior

   class DiscoveryMinimalBridge:
       """Minimal sampler bridge example."""

       def __init__(self, discovery_model, priors, latex_labels=None):
           self.model = discovery_model

           # Parse priors
           self.sampled_prior_dict, self.fixed_param_dict = _split_priors(priors)

           # Parameter names
           self.sampled_names = list(self.sampled_prior_dict.keys())
           self.fixed_names = list(self.fixed_param_dict.keys())
           self.discovery_paramnames = self.sampled_names + self.fixed_names

           # Counts
           self.n_sampled = len(self.sampled_names)
           self.n_fixed = len(self.fixed_names)

           # LaTeX labels
           self.latex_labels = latex_labels or {
               name: name for name in self.discovery_paramnames
           }

       def dict_to_array(self, params_dict):
           """Convert parameter dict to array."""
           return np.array([params_dict[name] for name in self.sampled_names])

       def array_to_dict(self, params_array):
           """Convert array to parameter dict (including fixed params)."""
           params_dict = {
               name: val for name, val in zip(self.sampled_names, params_array)
           }
           params_dict.update(self.fixed_param_dict)
           return params_dict

       def log_likelihood(self, params_array):
           """Evaluate log-likelihood from array."""
           params_dict = self.array_to_dict(params_array)
           return self.model(params_dict)

       def run_sampler(self, n_samples=1000):
           """Run a simple rejection sampler (example)."""
           # Sample from priors
           samples = []
           log_likes = []

           for _ in range(n_samples):
               # Sample from priors
               params_array = np.array([
                   self._sample_single_prior(self.sampled_prior_dict[name])
                   for name in self.sampled_names
               ])

               # Evaluate likelihood
               log_L = self.log_likelihood(params_array)

               samples.append(params_array)
               log_likes.append(log_L)

           return {
               'samples': np.array(samples),
               'log_likes': np.array(log_likes),
           }

       def _sample_single_prior(self, prior_spec):
           """Sample from single prior specification."""
           # Implement based on prior format
           pass

Reusing Common Components
^^^^^^^^^^^^^^^^^^^^^^^^^^

Import utility functions from existing interfaces:

.. code-block:: python

   from discoverysamplers.nessai_interface import (
       _split_priors,
       _parse_single_prior,
       ParsedPrior,
       _DiscoveryAdapter
   )

   class DiscoveryNewBridge:
       def __init__(self, model, priors, latex_labels=None):
           # Use existing prior parsing
           self.sampled_prior_dict, self.fixed_param_dict = _split_priors(priors)

           # Parse each prior
           self.parsed_priors = {
               name: _parse_single_prior(spec, name)
               for name, spec in self.sampled_prior_dict.items()
           }

           # Use discovery adapter
           self.adapter = _DiscoveryAdapter(
               model,
               self.fixed_param_dict,
               allow_array_api=True
           )

Adding New Prior Types
----------------------

Extend Prior Parsing
^^^^^^^^^^^^^^^^^^^^

To add new prior types, extend the prior parsing functions:

.. code-block:: python

   def _parse_single_prior_extended(prior_spec, param_name):
       """Extended prior parser with new types."""
       from discoverysamplers.nessai_interface import _parse_single_prior, ParsedPrior

       # First try standard parsing
       try:
           return _parse_single_prior(prior_spec, param_name)
       except:
           pass

       # Add new prior types
       if isinstance(prior_spec, dict):
           dist_type = prior_spec.get('dist', '')

           if dist_type == 'beta':
               # Beta distribution
               return ParsedPrior(
                   dist_type='beta',
                   a=prior_spec['a'],
                   b=prior_spec['b'],
                   bounds=(0, 1)
               )

           elif dist_type == 'gamma':
               # Gamma distribution
               return ParsedPrior(
                   dist_type='gamma',
                   shape=prior_spec['shape'],
                   scale=prior_spec.get('scale', 1.0),
                   bounds=(0, np.inf)
               )

       # If nothing worked, raise error
       raise ValueError(f"Cannot parse prior specification: {prior_spec}")

Custom Prior Container
^^^^^^^^^^^^^^^^^^^^^^

For samplers that need special prior objects:

.. code-block:: python

   class CustomPriorContainer:
       """Container for prior distributions."""

       def __init__(self, prior_dict):
           """
           Initialize from prior dictionary.

           Parameters
           ----------
           prior_dict : dict
               Dictionary mapping parameter names to prior specs
           """
           self.priors = {}

           for name, spec in prior_dict.items():
               self.priors[name] = self._create_prior(spec)

       def _create_prior(self, spec):
           """Create prior object from specification."""
           # Convert spec to appropriate prior object
           pass

       def sample(self, n=1):
           """Sample from all priors."""
           samples = {}
           for name, prior in self.priors.items():
               samples[name] = prior.sample(n)
           return samples

       def logpdf(self, params):
           """Compute log prior probability."""
           log_p = 0.0
           for name, value in params.items():
               log_p += self.priors[name].logpdf(value)
           return log_p

See Also
--------

- :doc:`../user_guide/model_requirements` - Model requirements
- :doc:`custom_priors` - Custom prior specifications
- Existing interfaces in ``src/discoverysamplers/`` for examples
