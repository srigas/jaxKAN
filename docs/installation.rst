.. _installation:

Installation
============

This section provides information for the installation of jaxKAN and its dependencies.

Requirements
------------

jaxKAN is built on JAX, which supports CPU functionality across all conventional platforms. However, GPU support is currently limited to Linux and is in an experimental stage for Windows WSL2. Consequently, the default CPU-based version of jaxKAN can be installed on all platforms, while GPU usage remains restricted due to the current limitations of JAX.

In addition, prior to installing jaxKAN, install:

- Python 3.11 or 3.12.
- all necessary NVIDIA drivers, if intending to use the GPU version of jaxKAN.


PyPI
----

jaxKAN is available as a PyPI package, so it can be installed using pip:

.. code-block:: bash

   pip install jaxkan
   
This will install the CPU version of the JAX dependency. For users intending to work with the GPU version (which is heavily recommended), install the package using the [gpu] option:

.. code-block:: bash

   pip install jaxkan[gpu]
   
To also install the dependencies required to run the :ref:`tutorials`, use the [doc] option:

.. code-block:: bash

   pip install jaxkan[doc]
   
   
Source
------

For pre-release versions of jaxKAN, clone the repository and build it from source:

.. code-block:: bash

   git clone https://github.com/srigas/jaxkan.git
   cd jaxkan
   pip install .

For additional dependencies, include the options mentioned in the PyPI section.