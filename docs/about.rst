.. _about:

About
=====

jaxKAN is a Python package designed to enable the training of Kolmogorov-Arnold Networks (KANs) using the JAX framework. Built on Flax's NNX module, jaxKAN provides a collection of KAN layers that serve as foundational building blocks for various KAN architectures, such as the EfficientKAN and the ChebyKAN. While it includes standard features like initialization and forward pass methods, the KAN class in jaxKAN introduces an `extend_grids` method, which facilitates the extension of the grids for all layers in the network, irrespective of how those grids are defined. For instance, in the case of ChebyKAN, where a traditional grid concept doesn't exist, the method extends the order of the Chebyshev polynomials utilized in the model.

Although KANs implemented in jaxKAN can be applied across a wide range of problem domains as a powerful alternative to Multilayer Perceptrons (MLPs), the package places a strong emphasis on their application in Physics-Informed Kolmogorov-Arnold Networks (PIKANs). To support this focus, jaxKAN includes specialized utilities and tutorials aimed at the task of solving forward or inverse PDE problems.

Research
---------

If you have used jaxKAN in your research, we'd love to hear from you! Below, you can find a list of academic publications that have used jaxKAN.

- A. A. Howard, B. Jacob, S. H. Murphy, A. Heinlein, P. Stinis, "Finite basis Kolmogorov-Arnold networks: domain decomposition for data-driven and physics-informed problems," `arXiv preprint`, 2024. https://doi.org/10.48550/arXiv.2406.19662

- S. Rigas, M. Papachristou, T. Papadopoulos, F. Anagnostopoulos and G. Alexandridis, "Adaptive Training of Grid-Dependent Physics-Informed Kolmogorov-Arnold Networks," in `IEEE Access`, vol. 12, pp. 176982-176998, 2024. https://doi.org/10.1109/ACCESS.2024.3504962

- A. A. Howard, B. Jacob, P. Stinis, "Multifidelity Kolmogorov-Arnold Networks," `arXiv preprint`, 2024. https://doi.org/10.48550/arXiv.2410.14764

- B. Jacob, A. A. Howard, P. Stinis, "SPIKANs: Separable Physics-Informed Kolmogorov-Arnold Networks," `arXiv preprint`, 2024. https://doi.org/10.48550/arXiv.2411.06286
