Physics-Informed Holomorphic Neural Networks (PIHNNs)
====================================================
<p align="center">
<a href="https://github.com/teocala/pihnn/actions"><img src="https://github.com/teocala/pihnn/actions/workflows/actions.yml/badge.svg" /></a>
<a href="https://github.com/teocala/pihnn"><img src="https://matteocalafa.com/badges/PIHNN-version.svg" /></a>
<a href="https://matteocalafa.com/PIHNN"><img src="https://matteocalafa.com/badges/PIHNN-doc.svg" /></a>
<a href="https://www.sciencedirect.com/science/article/pii/S0045782524006613"><img src="https://matteocalafa.com/badges/PIHNN-cite.svg" /></a>
<a href="https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html"><img src="https://matteocalafa.com/badges/PIHNN-license.svg" /></a>
</p>

PIHNNs can be used to efficiently learn solutions of PDEs exploiting their holomorphic representation.  

Author: [Matteo Calafà](https://matteocalafa.com/), research assistant at [Aarhus University](https://mpe.au.dk/en/).

Documentation
--------------
You can read about PIHNNs in our [paper](https://www.sciencedirect.com/science/article/pii/S0045782524006613).  
Furthermore, see [here](https://matteocalafa.com/PIHNN) the documentation of the library.

Installation
-------------
Clone the repository:
```
git clone https://github.com/teocala/pihnn.git
```
and move to the library main folder.  
Optionally, we provide an Anaconda environment with all dependencies; activate it by running
```
conda env create -f environment.yml
conda activate pihnn-env
```
Finally, install the library through pip:
```
pip install .
```

Getting started
---------------
The `examples` folder contains some tests that can be easily run from command line.  
For example, 
```
python3 examples/simply_connected/laplace.py
```
will run the test on the Laplace equation and provide plots in the `results` folder.


Citation
---------
If you use the library, please cite our work:
```
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
```
Tests conducted in the paper are also available in the `examples` folder.

License
-------------
[LGPL 2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
