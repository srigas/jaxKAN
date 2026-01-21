.. _about:

About
=====

jaxKAN is a Python package designed to enable the training of Kolmogorov-Arnold Networks (KANs) using the JAX framework. Built on Flax's NNX module, jaxKAN provides a collection of KAN layers that serve as foundational building blocks for various KAN architectures, such as the EfficientKAN and the ChebyKAN. While it includes standard features like initialization and forward pass methods, the KAN class in jaxKAN introduces an `extend_grids` method, which facilitates the extension of the grids for all layers in the network, irrespective of how those grids are defined. For instance, in the case of ChebyKAN, where a traditional grid concept doesn't exist, the method extends the order of the Chebyshev polynomials utilized in the model.

Although KANs implemented in jaxKAN can be applied across a wide range of problem domains as a powerful alternative to Multilayer Perceptrons (MLPs), the package places a strong emphasis on their application in Physics-Informed Kolmogorov-Arnold Networks (PIKANs). To support this focus, jaxKAN includes specialized utilities and tutorials aimed at the task of solving forward or inverse PDE problems.

The source code for jaxKAN can be found in the `jaxKAN GitHub Repository <https://github.com/srigas/jaxKAN>`_.


Research
---------

If you have used jaxKAN in your research, we'd love to hear from you! Below, you can find a list of academic publications that have used jaxKAN.

- Rigas, S., Anagnostopoulos, F., Papachristou, M., & Alexandridis, G. (2026). Training deep physics-informed Kolmogorov–Arnold networks. Computer Methods in Applied Mechanics and Engineering, 452, 118761. https://doi.org/10.1016/j.cma.2026.118761

- Cerardi, N., Tolley, E., & Mishra, A. (2026). Solving the cosmological Vlasov–Poisson equations with physics-informed Kolmogorov–Arnold networks. Monthly Notices of the Royal Astronomical Society, 545, staf2241. https://doi.org/10.1093/mnras/staf2241 | `GitHub Reference <https://github.com/nicolas-cerardi/cdm-pikan>`_

- Daniels, M., & Rigollet, P. (2025). Splat regression models (No. arXiv:2511.14042). arXiv. https://doi.org/10.48550/arXiv.2511.14042

- Rigas, S., Verma, D., Alexandridis, G., & Wang, Y. (2025). Initialization schemes for Kolmogorov-Arnold networks: An empirical study. arXiv. https://doi.org/10.48550/ARXIV.2509.03417 | `GitHub Reference <https://github.com/srigas/KAN_Initialization_Schemes>`_

- Howard, A. A., Jacob, B., & Stinis, P. (2025). Multifidelity kolmogorov–arnold networks. Machine Learning: Science and Technology, 6(3), 035038. https://doi.org/10.1088/2632-2153/adf702

- Jacob, B., Howard, A. A., & Stinis, P. (2025). SPIKANs: Separable physics-informed Kolmogorov–Arnold networks. Machine Learning: Science and Technology, 6(3), 035060. https://doi.org/10.1088/2632-2153/ae05af

- Howard, A. A., Jacob, B., Helfert, S., Heinlein, A., & Stinis, P. (2024). Finite basis Kolmogorov-Arnold networks: Domain decomposition for data-driven and physics-informed problems. arXiv. https://doi.org/10.48550/ARXIV.2406.19662

- Rigas, S., Papachristou, M., Papadopoulos, T., Anagnostopoulos, F., & Alexandridis, G. (2024). Adaptive training of grid-dependent physics-informed kolmogorov-arnold networks. IEEE Access, 12, 176982–176998. https://doi.org/10.1109/ACCESS.2024.3504962 | `GitHub Reference <https://github.com/srigas/jaxKAN>`_
