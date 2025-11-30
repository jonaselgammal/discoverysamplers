JAX-NS Interface
================

This module provides the bridge interface connecting Discovery models to the
`JAX-NS <https://github.com/Joshuaalbert/jaxns>`_ nested sampler. JAX-NS is a
pure JAX implementation that supports GPU acceleration and vectorized likelihood
evaluation.

.. hint::

   For best performance with JAX-NS, ensure your likelihood is JAX-compatible
   and enable JIT compilation with ``jit=True``.

.. automodule:: discoverysamplers.jaxns_interface
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`../user_guide/jaxns_usage` - User guide for JAX-NS
- :doc:`../advanced/performance` - Performance optimization
