.. discoverysamplers documentation master file

Welcome to discoverysamplers documentation!
============================================

**discoverysamplers** is a collection of bridge interfaces that connect Discovery (a JAX-based Pulsar Timing Array analysis framework) to various nested sampling and MCMC samplers. The package provides lightweight wrappers that adapt Discovery-style models to the APIs expected by different sampling backends.

Key Features
------------

- **Unified Interface**: Consistent API across multiple sampling backends
- **Flexible Prior Specification**: Support for multiple prior formats (dicts, tuples, callables)
- **JAX Integration**: Optional JIT compilation for improved performance
- **Multiple Samplers**: Support for Eryn (MCMC), Nessai (flow-based nested sampling), JAX-NS, and GPry (GP emulation)
- **Parameter Management**: Automatic handling of fixed vs. sampled parameters
- **LaTeX Labels**: Built-in support for publication-quality plotting

Supported Samplers
------------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Sampler
     - Type
     - Key Features
   * - **Eryn**
     - MCMC (Ensemble)
     - Parallel tempering, reversible-jump MCMC support
   * - **Nessai**
     - Nested Sampling
     - Flow-based proposals, importance nested sampling
   * - **JAX-NS**
     - Nested Sampling
     - Pure JAX implementation, vectorized likelihood evaluation
   * - **GPry**
     - GP Emulation
     - Gaussian Process surrogate model with active learning, accelerates expensive likelihoods

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install discoverysamplers

Basic usage with Nessai:

.. code-block:: python

   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge

   # Define your Discovery model
   def my_model(params):
       # Your likelihood calculation
       return log_likelihood

   # Define priors
   priors = {
       'mass': ('uniform', 1.0, 3.0),
       'distance': ('loguniform', 0.1, 100.0),
       'phase': ('uniform', 0, 2*np.pi),
   }

   # Create bridge and run sampler
   bridge = DiscoveryNessaiBridge(my_model, priors)
   results = bridge.run_sampler(nlive=1000, output='output_directory/')

See the :doc:`user_guide/quickstart` for more detailed examples.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/prior_specification
   user_guide/eryn_usage
   user_guide/nessai_usage
   user_guide/jaxns_usage
   user_guide/gpry_usage
   user_guide/model_requirements

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/priors
   api/likelihood
   api/eryn_interface
   api/nessai_interface
   api/jaxns_interface
   api/gpry_interface
   api/eryn_rj_interface

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/performance
   advanced/parallel_tempering
   advanced/reversible_jump
   advanced/custom_priors
   advanced/extending

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/notebooks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
