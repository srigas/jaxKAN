---
title: 'jaxKAN: A unified JAX framework for Kolmogorov-Arnold Networks'
tags:
  - Python
  - JAX
  - Kolmogorov-Arnold Networks
  - Physics-Informed Neural Networks
  - PIKANs
authors:
  - name: Spyros Rigas
    orcid: 0009-0009-2352-8709
	corresponding: true
    affiliation: 1
  - name: Michalis Papachristou
    orcid: 0000-0001-5650-4206
    affiliation: 2
affiliations:
 - name: Department of Digital Industry Technologies, School of Science, National and Kapodistrian University of Athens
   index: 1
 - name: Department of Physics, School of Science, National and Kapodistrian University of Athens
   index: 2
date: 8 January 2024
bibliography: paper.bib
---

# Summary

`jaxKAN` is a JAX-based library for building and training Kolmogorov–Arnold Networks (KANs) [@Liu:2024],
built on Flax's NNX [@flax] with Optax [@optax] for optimization. It provides a broad selection of layer
implementations - from the original KAN design to more recent or efficient variants - and unifies them under
a single interface. Beyond basic model instance definition and training, `jaxKAN` supports class-inherent
adaptive training methods (e.g., grid updates) and provides utilities that address performance limitations in
the original KAN framework. KANs from `jaxKAN` can be used in any setting where standard multilayer perceptrons (MLPs)
would otherwise be employed as the underlying architecture, although the library includes specialized utilities
for adaptive Partial Differential Equation (PDE) solving tasks, thus placing emphasis on scientific applications
with Physics-Informed Kolmogorov-Arnold Networks (PIKANs) [@Karniadakis:2024].

# Statement of need

The recent introduction of KANs [@Liu:2024] in machine learning provided an alternative to traditional MLPs
by making the network activation functions trainable, through expansions in terms of basis functions - originally
B-splines. This novel idea proved to be effective across diverse applications, from engineering to scientific
computing. Consequently, KANs were extended to incorporate alternative basis functions, such as
polynomial [@Sidharth:2024] or trigonometric [@fkan] functions. Moreover, more efficient architectures were
proposed to overcome performance bottlenecks in the original design [@effkan]. As a result, a growing body of
research has explored new ways to build, refine and utilize KANs, yet a comprehensive, high-performance
software framework that integrates these ideas under one roof has been lacking.

Existing libraries only partially address this need. For instance, `pykan` [@pykan], the original
`PyTorch`-based [@paszke:2019] KAN implementation, does not provide expandability to multiple layer types.
Similarly, frameworks like `NeuroMANCER` [@neuromancer], while valuable for physics-informed applications,
restrict KAN usage mainly to scientific computing tasks via the use of high-level APIs, thus limiting their
applicability for other types of problems or low-level integration into novel deep learning architectures.
Additionally, while decentralized repositories exist for individual KAN layer types, there is no unified
approach to constructing and training KANs, making it challenging to combine them into hybrid architectures or
incorporate them into broader workflows. This fragmented landscape perpetuates the tendency towards assembling
custom, case-specific solutions by researchers and practitioners.

`jaxKAN` fills this gap by providing a JAX-native codebase that encompasses multiple KAN layer types, from the
original B-spline-based approach to alternative variants. Through its unified interface, users can not only 
choose from various basis expansions but also utilize built-in adaptive routines - such as grid updates,
adaptive state transitions, or basis order extension - to tackle problems where static representations
underperform. Moreover, it delivers specialized utilities for the adaptive training of PIKANs, a field where KANs
have proven to be equally or even more efficient than MLPs in several occasions [@Karniadakis:2024; @Rigas:2024].
At the time of writing, `jaxKAN` has been used in a series of academic works relevant to PDE
problems [@Howard1:2024; @Rigas:2024; @Howard2:2024; @Jacob:2024], also showcasing significantly
faster performance compared to the original `pykan` implementation [@Rigas:2024].

# References