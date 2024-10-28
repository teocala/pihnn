PIHNN 1.0.0
=============

This library provides the implementation of physics-informed holomorphic neural networks for the Laplace, biharmonic and plane linear elasticity equations.

You can read about PIHNNs in `Calafà et al. [2024] <https://www.sciencedirect.com/science/article/pii/S0045782524006613>`_.
Alternatively, you can watch the `video <https://www.youtube.com/watch?v=37mDjIVfSho&t=1665s>`_ of our presentation at the CRUNCH seminar. 

Before using the library, we recommend to read here the documentation and look at some applications in the ``examples`` folder.

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

Getting started
---------------
The ``examples`` folder contains some tests that can be easily run from command line.  
For example, 

.. code-block:: bash

   python3 examples/simply_connected/laplace.py

will run the test on the Laplace equation and provide plots in the ``results`` folder.


Using CUDA 
-----------
It is possible to exploit GPUs for accelerated processing if you have a `CUDA-capable <https://developer.nvidia.com/cuda-zone>`_ system with the `CUDA toolkit <https://developer.nvidia.com/cuda-zone>`_ installed. 
Furthermore, you need to install the `CUDA enabled PyTorch <https://pytorch.org/get-started/locally/>`_ by running:

.. code-block:: bash

   conda install pytorch-cuda -c pytorch -c nvidia

Cite
-----------

If you use the library, don't forget to cite our work:

.. code-block:: bib

   @article{CALAFA2024117406,
         title = {Physics-Informed Holomorphic Neural Networks ({PIHNNs}): Solving {2D} linear elasticity problems},
         author = {Matteo Calafà and Emil Hovad and Allan P. Engsig-Karup and Tito Andriollo},
         journal = {Computer Methods in Applied Mechanics and Engineering},
         volume = {432},
         pages = {117406},
         year = {2024},
         issn = {0045-7825},
         doi = {https://doi.org/10.1016/j.cma.2024.117406}
   }

License
-------------
`LGPL 2.1 <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>`_


Modules
------------

.. toctree::
   :maxdepth: 1

   nn.rst
   utils.rst
   geometries.rst
   graphics.rst
   bc.rst