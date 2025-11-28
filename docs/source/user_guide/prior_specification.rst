Prior Specification
===================

Priors are provided as a single dictionary mapping parameter names to distributions. When you pass a Discovery model (e.g., ``likelihood`` or ``likelihood.logL``), the keys **must match** the names exposed by that model (``likelihood.logL.params``). Fixed values can live in the same dictionary as sampled parameters.

**Note**: All sampler bridges accept either ``likelihood`` or ``likelihood.logL`` — they automatically extract the ``logL`` callable if you pass the whole object.

Quick example (from the :doc:`quickstart` pulsar setup):

.. code-block:: python

   import discovery as ds
   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

   psr = ds.Pulsar.read_feather("path/to/pulsar.feather")
   likelihood = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),
   ])

   priors = {
       "x": ("uniform", -5.0, 5.0),
       "y": ("loguniform", 0.1, 10.0),
       "fixed_param": ("fixed", 1.0),
   }

   bridge = DiscoveryNessaiBridge(discovery_model=likelihood, priors=priors)

Base Prior Shapes
-----------------

The common parser in ``discoverysamplers.priors`` understands these shapes (dict and tuple forms are equivalent):

- ``("uniform", min, max)`` or ``{"dist": "uniform", "min": a, "max": b}``
- ``("loguniform", min, max)`` or ``{"dist": "loguniform", "min": a, "max": b}``
- ``("normal", mean, sigma[, min, max])`` or ``{"dist": "normal", "mean": m, "sigma": s, "min": a, "max": b}`` (bounds optional unless noted below)
- ``("fixed", value)`` or ``{"dist": "fixed", "value": v}`` or simply ``v`` (scalar shorthand)
- **Callable** objects with ``logpdf(value)`` and optional ``bounds`` attribute

Sampler-Specific Support
------------------------

Nessai (nested sampling)
^^^^^^^^^^^^^^^^^^^^^^^^
- Supports: uniform, loguniform, normal (requires finite ``min``/``max``), fixed, callable with ``logpdf`` and ``bounds``.
- Implementation: ``_split_priors_nessai`` enforces finite bounds; unbounded normals raise ``PriorParsingError``.
- Outside the specified bounds, log priors return ``-inf`` (see :mod:`discoverysamplers.nessai_interface`).

JAX-NS (nested sampling)
^^^^^^^^^^^^^^^^^^^^^^^^
- Supports: uniform, loguniform, normal (optionally truncated), fixed.
- Implementation: the prior transform in :mod:`discoverysamplers.jaxns_interface` builds unit-cube transforms for these three distributions; callable priors are not supported for sampling.
- Provide finite bounds for stability; unbounded normals are technically allowed but avoid them for nested sampling.

GPry via Cobaya
^^^^^^^^^^^^^^^
- Supports: uniform (``min``/``max``), loguniform (``a``/``b`` in SciPy/Cobaya style), normal (mean/sigma, optionally add ``min``/``max`` to truncate), fixed.
- Implementation: :mod:`discoverysamplers.gpry_interface` converts parsed priors to Cobaya ``params`` entries; callable priors are not accepted here.
- Bounds control the support used in Cobaya/GPry; outside the provided interval the helper log-priors return ``-inf``.

Eryn (MCMC)
^^^^^^^^^^^
- Supports: uniform (``min``/``max``), loguniform (keys ``a``/``b``), fixed values, or any object exposing ``logpdf`` and ``rvs``.
- Implementation: :mod:`discoverysamplers.eryn_interface` wraps these in ``ProbDistContainer``; defaults can be taken from Discovery if priors are set to ``None``/``"default"``.
- No explicit bounds are required for MCMC, but you should still set sensible ranges to help initialization.
- **RJMCMC mode**: Uses a different branch-indexed format; see the `Reversible Jump MCMC Priors`_ section below.

Custom Priors
-------------

Provide any object with a ``logpdf`` method and (optionally) a ``bounds`` tuple. The sampler will evaluate the log prior only inside the bounds and return ``-inf`` outside.

.. code-block:: python

   class BoxPrior:
       def __init__(self, low, high):
           self.bounds = (low, high)
       def logpdf(self, x):
           if self.bounds[0] <= x <= self.bounds[1]:
               return -np.log(self.bounds[1] - self.bounds[0])
           return -np.inf

   priors = {
       "param": BoxPrior(0.0, 10.0),  # Works for Nessai (with bounds), JAX-NS (not supported), Eryn, GPry (not supported)
   }

Reversible Jump MCMC Priors
---------------------------

For trans-dimensional sampling with Eryn's RJMCMC mode, priors have a different structure. Instead of mapping parameter names to distributions, you map **branch names** to dictionaries indexed by component count:

.. code-block:: python

   from discoverysamplers.eryn_RJ_interface import DiscoveryErynRJBridge

   # RJMCMC prior format: {branch_name: {n_components: prior_dict, ...}, ...}
   rj_priors = {
       'cw': {
           0: {},  # Empty dict for 0 components (no parameters)
           1: {
               'cw_f': ('loguniform', 1e-9, 1e-7),
               'cw_h': ('loguniform', 1e-20, 1e-14),
               'cw_phi': ('uniform', 0, 2*np.pi),
           },
           2: {
               'cw_f': ('loguniform', 1e-9, 1e-7),
               'cw_h': ('loguniform', 1e-20, 1e-14),
               'cw_phi': ('uniform', 0, 2*np.pi),
           },
           # ... more component counts as needed
       }
   }

   bridge = DiscoveryErynRJBridge(
       discovery_model=rj_model,
       priors=rj_priors,
       branch_names=['cw'],
       nleaves_min={'cw': 0},  # Minimum components per branch
       nleaves_max={'cw': 5},  # Maximum components per branch
   )

**Key differences from fixed-dimensional priors**:

1. **Branch-indexed**: Top-level keys are branch names (e.g., ``'cw'`` for continuous wave sources)
2. **Component-indexed**: Within each branch, keys are integers (0, 1, 2, ...) representing component counts
3. **Per-component priors**: Each component count maps to a regular prior dictionary
4. **Zero components**: Include an empty dict ``{}`` for 0 components (model with no sources)

**Model compatibility**: Your RJMCMC model's ``logL`` method must accept parameters in a nested list format::

   # Model receives parameters as nested lists
   def logL(self, *params):
       # params[i] is a list of arrays, one per component
       # params[i][j] has shape (nwalkers, 1) for component j
       cw_params = params[0]  # First branch parameters
       n_components = len(cw_params)
       ...

See :doc:`eryn_usage` for complete RJMCMC examples.

Bounded Priors for GPry and Nested Sampling
-------------------------------------------

Nested samplers and GPry operate on bounded spaces:

- **Nessai** refuses infinite bounds; every sampled parameter must supply finite ``min``/``max`` (or callable ``bounds``), otherwise initialization raises ``PriorParsingError``.
- **JAX-NS** builds its unit-cube transform from the provided bounds; supply finite ranges for uniform/loguniform and truncate normals when possible to keep the transform well behaved.
- **GPry/Cobaya** expects bounded priors; uniform and loguniform require explicit limits, and you can cap normals with ``min``/``max`` to keep the surrogate within a finite region.
- The helper log-priors clamp evaluation to whatever bounds are available (explicit ``min``/``max`` or a callable's ``bounds``), returning ``-inf`` outside. For nested samplers and GPry this automatic capping keeps the likelihood inside a finite region, so make sure those bounds are present.

See Also
--------

- :doc:`model_requirements` - Requirements for Discovery models
- :doc:`quickstart` - Basic usage examples
- :doc:`../advanced/custom_priors` - Advanced prior customization
