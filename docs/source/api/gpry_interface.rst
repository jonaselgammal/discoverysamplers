GPry Interface
==============

This module provides the bridge interface connecting Discovery models to
`GPry <https://github.com/jonaselgammal/GPry>`_ via the Cobaya framework.
GPry uses Gaussian Process emulation with active learning to efficiently
explore the parameter space.

.. hint::

   GPry is particularly useful for expensive likelihoods where each evaluation
   is costly. The GP surrogate model learns the likelihood surface with fewer
   evaluations than traditional samplers.

.. automodule:: discoverysamplers.gpry_interface
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`../user_guide/gpry_usage` - User guide for GPry
- `Cobaya documentation <https://cobaya.readthedocs.io/>`_ - Cobaya framework
- `GPry documentation <https://gpry.readthedocs.io/>`_ - GPry package
