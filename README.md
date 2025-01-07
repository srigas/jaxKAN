[![Doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://jaxkan.readthedocs.io/)
[![License](https://img.shields.io/github/license/srigas/jaxkan)](https://github.com/srigas/jaxKAN/blob/main/LICENSE)
[![Run Tests](https://github.com/srigas/jaxKAN/actions/workflows/test.yml/badge.svg)](https://github.com/srigas/jaxKAN/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/jaxkan.svg)](https://pypi.org/project/jaxkan/)

# jaxKAN

jaxKAN is a Python package designed to enable the training of Kolmogorov-Arnold Networks (KANs) using the JAX framework. Built on Flax's NNX module, jaxKAN provides a collection of KAN layers that serve as foundational building blocks for various KAN architectures, such as the EfficientKAN and the ChebyKAN. While it includes standard features like initialization and forward pass methods, the KAN class in jaxKAN introduces an `extend_grids` method, which facilitates the extension of the grids for all layers in the network, irrespective of how those grids are defined. For instance, in the case of ChebyKAN, where a traditional grid concept doesn't exist, the method extends the order of the Chebyshev polynomials utilized in the model.


## Documentation

Extensive documentation on jaxKAN, including installation & contributing guidelines, API reference and tutorials, can be found [here](https://jaxkan.readthedocs.io/).


## Citation

There is a JOSS paper currently submitted under review for jaxKAN. Until it is published, if you utilized `jaxKAN` for your own academic work, please consider using the following citation, which is the paper in which the framework was first introduced for PIKANs:

```
@article{10763509,
      author = {Rigas, Spyros and Papachristou, Michalis and Papadopoulos, Theofilos and Anagnostopoulos, Fotios and Alexandridis, Georgios},
      journal = {IEEE Access}, 
      title = {Adaptive Training of Grid-Dependent Physics-Informed Kolmogorov-Arnold Networks}, 
      year = {2024},
      volume = {12},
      pages = {176982-176998},
      doi = {10.1109/ACCESS.2024.3504962}
}
```
