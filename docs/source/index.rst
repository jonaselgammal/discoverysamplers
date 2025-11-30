.. discoverysamplers documentation master file

Welcome to discoverysamplers documentation!
============================================

**discoverysamplers** is a collection of bridge interfaces that connect `Discovery <https://github.com/nanograv/discovery>`_ (a JAX-based Pulsar Timing Array analysis framework) to various nested sampling and MCMC samplers.

.. list-table::
   :widths: 30 70

   * - **Author**
     - Jonas El Gammal
   * - **Source**
     - `Source code on GitHub <https://github.com/jonaselgammal/discoverysamplers>`_
   * - **Documentation**
     - `Documentation on Read the Docs <https://discoverysamplers.readthedocs.io>`_
   * - **License**
     - `MIT <https://opensource.org/licenses/MIT>`_
   * - **Support**
     - For questions, use `GitHub Issues <https://github.com/jonaselgammal/discoverysamplers/issues>`_ or drop me an email
   * - **Installation**
     - Clone from `GitHub <https://github.com/jonaselgammal/discoverysamplers>`_

Overview
--------

The package provides lightweight wrappers that adapt `Discovery <https://github.com/nanograv/discovery>`_-style models (callables accepting parameter dictionaries) to the APIs expected by different sampling backends.

Key Features
------------

- **Unified Interface**: Consistent API across multiple sampling backends
- **Flexible Prior Specification**: Support for multiple prior formats (dicts, tuples, callables)
- **JAX Integration**: Optional JIT compilation for improved performance
- **Parameter Management**: Automatic handling of fixed vs. sampled parameters
- **LaTeX Labels**: Built-in support for publication-quality plotting

Supported Samplers
------------------

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Sampler
     - Type
     - Key Features
   * - `Eryn <https://github.com/mikekatz04/Eryn>`_
     - MCMC (Ensemble)
     - Parallel tempering, reversible-jump MCMC support
   * - `Nessai <https://github.com/mj-will/nessai>`_
     - Nested Sampling
     - Flow-based proposals, importance nested sampling
   * - `JAX-NS <https://github.com/Joshuaalbert/jaxns>`_
     - Nested Sampling
     - Pure JAX implementation, vectorized likelihood evaluation
   * - `GPry <https://github.com/jonaselgammal/GPry>`_
     - GP Emulation
     - Gaussian Process surrogate model with active learning

Quick Start
-----------

Install the package:

.. code-block:: bash

   git clone https://github.com/jonaselgammal/discoverysamplers.git
   cd discoverysamplers
   pip install .

Basic usage with Nessai:

.. code-block:: python

   from discoverysamplers import DiscoveryNessaiBridge

   # Define your model
   def my_model(params):
       return -0.5 * (params['x']**2 + params['y']**2)

   # Define priors
   priors = {
       'x': ('uniform', -5.0, 5.0),
       'y': ('uniform', -5.0, 5.0),
   }

   # Create bridge and run sampler
   bridge = DiscoveryNessaiBridge(my_model, priors)
   results = bridge.run_sampler(nlive=1000, output='output/')

See the :doc:`user_guide/quickstart` for more detailed examples.

Citation
--------

If you use ``discoverysamplers`` in your research, please cite:

.. code-block:: bibtex

   @software{discoverysamplers,
     author = {El Gammal, Jonas},
     title = {discoverysamplers: Tools for Bayesian inference with Discovery},
     year = {2025},
     url = {https://github.com/jonaselgammal/discoverysamplers}
   }

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
   user_guide/plotting
   user_guide/model_requirements

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

.. toctree::
   :maxdepth: 1
   :caption: Acknowledgements

   acknowledgements

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/priors
   api/likelihood
   api/plots
   api/eryn_interface
   api/nessai_interface
   api/jaxns_interface
   api/gpry_interface
   api/eryn_rj_interface

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
