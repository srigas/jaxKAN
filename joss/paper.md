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
a single interface. Beyond basic model instance definition and training, jaxKAN supports class-inherent
adaptive training methods (e.g., grid updates) and provides utilities that address performance limitations in
the original KAN framework. KANs from `jaxKAN` can be used in any setting where standard multilayer perceptrons (MLPs)
would otherwise be employed as the underlying architecture, although the library includes specialized utilities
for adaptive PDE solving tasks, thus placing emphasis on scientific applications with Physics-Informed
Kolmogorov-Arnold Networks (PIKANs) [@Karniadakis:2024].

# Statement of need

TODO

# References