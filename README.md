[![DOI](https://joss.theoj.org/papers/10.21105/joss.07830/status.svg)](https://doi.org/10.21105/joss.07830)
[![Doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://jaxkan.readthedocs.io/)
[![License](https://img.shields.io/github/license/srigas/jaxkan)](https://github.com/srigas/jaxKAN/blob/main/LICENSE)
[![Run Tests](https://github.com/srigas/jaxKAN/actions/workflows/test.yml/badge.svg)](https://github.com/srigas/jaxKAN/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/jaxkan.svg)](https://pypi.org/project/jaxkan/)

# jaxKAN

jaxKAN is a Python package designed to enable the training of Kolmogorov-Arnold Networks (KANs) using the JAX framework. Built on Flax's NNX module, jaxKAN provides a collection of KAN layers that serve as foundational building blocks for various KAN architectures, such as the EfficientKAN and the ChebyKAN. While it includes standard features like initialization and forward pass methods, the KAN class in jaxKAN introduces an `extend_grids` method, which facilitates the extension of the grids for all layers in the network, irrespective of how those grids are defined. For instance, in the case of ChebyKAN, where a traditional grid concept doesn't exist, the method extends the order of the Chebyshev polynomials utilized in the model.


## Documentation

Extensive documentation on jaxKAN, including installation & contributing guidelines, API reference and tutorials, can be found [here](https://jaxkan.readthedocs.io/).


## Contributing

We warmly welcome community contributions to jaxKAN! For details on the types of contributions that will help jaxKAN evolve, as well as guidelines on how to contribute, visit [this](https://jaxkan.readthedocs.io/en/latest/contributing.html) page of our documentation.


## Citation

If you utilized `jaxKAN` for your own academic work, please use the following citation:

```
@article{Rigas2025,
      author = {Rigas, Spyros and Papachristou, Michalis},
      title = {jax{KAN}: A unified {JAX} framework for {K}olmogorov-{A}rnold Networks},
      journal = {Journal of Open Source Software},
      year = {2025},
      volume = {10},
      number = {108},
      pages = {7830},
      doi = {10.21105/joss.07830}
}
```

If you have used jaxKAN in your research for PIKAN-related applications or theoretical developments, please consider also citing the paper that originally introduced jaxKAN for these tasks:

```
@article{10763509,
      author = {Rigas, Spyros and Papachristou, Michalis and Papadopoulos, Theofilos and Anagnostopoulos, Fotios and Alexandridis, Georgios},
      title = {Adaptive Training of Grid-Dependent Physics-Informed {K}olmogorov-{A}rnold Networks}, 
      journal = {IEEE Access},
      year = {2024},
      volume = {12},
      pages = {176982-176998},
      doi = {10.1109/ACCESS.2024.3504962}
}
```
