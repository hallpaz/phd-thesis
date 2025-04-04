# \[phd-thesis\] Multiresolution Spectral Representation of Neural Media

#### Thesis outline

## 1 Introduction

- General context of machine and deep learning in visual computing.
- Deep learning as a mean for media representation.
- Vision of neural networks based standards that are compact, fast to evaluate (TPU), and allow new operations or better operations with media (even if in specific domains).
- There are some works on this direction already, however there's a gap in understanding sinusoidal neural networks and representing media in multiresolution. Also, many works demonstrate proof of concepts to represent a full scene, but in the envisioned scenario, we will need primitives that can be edited and manipulated to compose more sophisticated scenes and use cases.

## 2 Theoretical Background

- Paradigm of the 4 universes as a concept to understand real/abstract world translation into media representation in a computer.
- Sampling and Reconstruction theory for discretization of the signals of the world abd understanding of the current most used representation that we have. Also, explanation of the limits and conditions for it to work correctly, avoiding commom pitfall as aliasing.
- Scale space theory: intimately associated with sampling is the ideia of resolution and filtering.
- (*Functional representation of signals*) Explain how we can represent images as a vector field of colors for a rectangular domain. More general, how we could have a vector field of attributes to represent any signal in space.
- Present the *implicit representation* as a way to extend this functional representation for "less obvious" cases. Implicit function theorem and applications to 3D models representation, for examples.
- Explain some advantages of a continuous representation like this over the discrete ones. Also, discuss its drawbacks.
- Explain how the term *implicit* has been overloaded in the deep learning visual computing community.
- Neural Networks and Sinusoidal neural networks

## 3 Representational Neural Networks

- Extend the concept of graphical object to media object and characterize it mathematically, showing how it is clear that neural networks can be used to encode media objects.
- Present examples of the application of this framework in understanding different kinds of media.

## 4 Multiresolution Sinusoidal Neural Networks

- Present the experiments in understanding the control of frequencies in a sinusoidal network:
    - Fitting a signal with few frequencies
    - Fitting a stochastic signal
    - Filtering a stochastic signal
    - Diverging by using high frequencies
- Present the experiments in fitting multiresolution signals
    - 1D Gaussian Tower
    - 1D Gaussian Pyramid
    - show that networks is well behaved between samples
- Present the MR-Net architecture
    - Discussion on Filtering, scheduling etc
    - show the atoms

## 5 Multiresolution Imaging
    - Discuss the functional explicit representation of images
    - Present imaging experiments
    - Comparison against Siren and Bacon
    - Kodak Dataset
    - Show a few examples in details

## 6 Textures
    - Present the math for the periodic spectral networks
    - Show how new frequencies are generated from a small and sparse set of frequencies
    - applications on tileable textures
    - 

## Interesting directions

# 7 Conclusions


