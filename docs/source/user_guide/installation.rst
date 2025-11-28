Installation
============

Basic Installation
------------------

``discoverysamplers`` is currently not available on PyPI. Install directly from the repository:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/discoverysamplers.git
   cd discoverysamplers

   # Install the package
   pip install .

This installs the core package without any sampler dependencies.

Installing with Sampler Dependencies
-------------------------------------

Since different samplers have different dependencies, you can install only the samplers you need.

First, install ``discoverysamplers`` as shown above, then install the specific samplers:

Eryn (MCMC)
^^^^^^^^^^^

.. code-block:: bash

   pip install eryn>=1.2

Nessai (Nested Sampling)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install nessai

JAX-NS (JAX-based Nested Sampling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install jaxns jax

GPry (via Cobaya)
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install gpry cobaya

Install All Samplers
^^^^^^^^^^^^^^^^^^^^

To install all supported samplers at once:

.. code-block:: bash

   pip install eryn nessai jaxns gpry cobaya jax

Development Installation
------------------------

For development, install in editable mode:

.. code-block:: bash

   git clone https://github.com/yourusername/discoverysamplers.git
   cd discoverysamplers
   pip install -e .

This creates an editable installation that reflects changes to the source code immediately.

Requirements
------------

Core Requirements
^^^^^^^^^^^^^^^^^

- Python >= 3.9
- NumPy (for array operations)

Optional Requirements
^^^^^^^^^^^^^^^^^^^^^

Depending on which sampler interfaces you plan to use:

- **eryn** >= 1.2 - For :class:`~discoverysamplers.eryn_interface.DiscoveryErynBridge`
- **nessai** - For :class:`~discoverysamplers.nessai_interface.DiscoveryNessaiBridge`
- **jaxns** - For :class:`~discoverysamplers.jaxns_interface.DiscoveryJAXNSBridge`
- **gpry** - For :class:`~discoverysamplers.gpry_interface.DiscoveryGPryCobayaBridge`
- **cobaya** - Required for GPry interface
- **jax** - Recommended for performance (used by JAX-NS and optionally by other interfaces)

Verifying Installation
----------------------

You can verify your installation by importing the package:

.. code-block:: python

   import discoverysamplers
   from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
   print("Installation successful!")

To check which samplers are available:

.. code-block:: python

   # Try importing each interface
   interfaces = []

   try:
       from discoverysamplers.eryn_interface import DiscoveryErynBridge
       interfaces.append("Eryn")
   except ImportError:
       pass

   try:
       from discoverysamplers.nessai_interface import DiscoveryNessaiBridge
       interfaces.append("Nessai")
   except ImportError:
       pass

   try:
       from discoverysamplers.jaxns_interface import DiscoveryJAXNSBridge
       interfaces.append("JAX-NS")
   except ImportError:
       pass

   try:
       from discoverysamplers.gpry_interface import DiscoveryGPryCobayaBridge
       interfaces.append("GPry")
   except ImportError:
       pass

   print(f"Available interfaces: {', '.join(interfaces)}")

Troubleshooting
---------------

JAX Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^

If you encounter issues installing JAX, refer to the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for platform-specific instructions, especially for GPU support.

Dependency Conflicts
^^^^^^^^^^^^^^^^^^^^

If you experience dependency conflicts between samplers, consider using separate virtual environments for different sampling workflows:

.. code-block:: bash

   # Environment for Nessai
   python -m venv venv_nessai
   source venv_nessai/bin/activate
   pip install discoverysamplers nessai

   # Environment for JAX-NS
   python -m venv venv_jaxns
   source venv_jaxns/bin/activate
   pip install discoverysamplers jaxns jax

Next Steps
----------

Once installed, proceed to the :doc:`quickstart` guide to learn how to use the package.
