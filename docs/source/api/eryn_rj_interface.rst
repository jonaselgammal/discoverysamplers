Eryn Reversible-Jump Interface
==============================

This module provides the bridge interface for reversible-jump MCMC (RJMCMC) using
`Eryn <https://github.com/mikekatz04/Eryn>`_. RJMCMC enables trans-dimensional
sampling where the number of model components can vary during sampling.

.. hint::

   RJMCMC is useful for model selection problems where you want to determine
   the optimal number of signal components (e.g., how many gravitational wave
   sources are present in the data).

.. automodule:: discoverysamplers.eryn_RJ_interface
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`../user_guide/eryn_usage` - User guide for Eryn (includes RJMCMC section)
- :doc:`../advanced/reversible_jump` - Advanced RJMCMC guide
- :doc:`../advanced/parallel_tempering` - Parallel tempering (recommended for RJMCMC)
- :doc:`eryn_interface` - Standard fixed-dimensional Eryn interface
