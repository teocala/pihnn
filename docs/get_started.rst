Get Started
=============

Installation
-------------

Clone the library source code:

.. code-block:: bash

   git clone https://github.com/teocala/pihnn.git

and move to the library main folder.

.. note::
   Before the installation, it is recommended to create an `Anaconda <https://anaconda.org/>`_ environment:

   .. code:: bash

      conda env create -f environment.yml # the file 'environment.yml' already provides all dependencies
      conda activate pihnn-env # 'pihnn-env' is the name of the newly created environment

Finally, run

.. code-block:: bash

   pip install .

.. warning::
   Make sure the name and version of the library are correctly detected during installation. There are currently some issues with some versions of pip and linux (e.g.,  `here <https://github.com/pypa/setuptools/issues/3269>`_)

Run a test
---------------
The ``examples`` folder contains some tests that can be easily run from command line.  
For example, 

.. code-block:: bash

   python3 examples/simply_connected/laplace.py

will run the test on the Laplace equation and provide plots in the ``results`` folder.


Using CUDA 
-----------
The library supports CUDA processing. If your system is `CUDA-capable <https://developer.nvidia.com/cuda-zone>`_, the code will automatically run on the GPU.


Citation
---------- 
If you use PIHNNs for your research, don't forget to cite our works in :doc:`../../publications`.
