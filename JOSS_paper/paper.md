---
title: 'IAMReX: A Multiphase Incompressible Flow Solver Based on the Immersed Boundary Method'
tags:
  - C++
  - Computational Fluid Dynamics
  - Immersed Boundary Method
  - Multiphase Flow
  - Adaptive Mesh
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 November 2024
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The IAMReX repository is constructed based on the IAMR code, dedicated to solving the equations of multiphase incompressible flows. It utilizes the projection method to solve the Navier-Stokes equations on a semi-staggered grid, an approach that effectively manages the dynamics of incompressible fluids. Regarding the capture of gas-liquid interfaces, IAMReX offers both the Level Set (LS) method and the Conservative Level Set (CLS) method; these two techniques can accurately capture the dynamic changes of gas-liquid interfaces, making them particularly suitable for problems involving free surface flows or multiphase flows. For fluid-solid interface issues, IAMReX employs the Multi-Direction Forcing Immersed Boundary Method (IBM), a powerful tool for simulating the interaction between fluids and complex solid boundaries without the need for grid matching on the solid boundaries, thereby reducing computational costs and increasing flexibility. Additionally, IAMReX captures collisions between particles and walls as well as between particles themselves through the Adaptive Collision Time Model (ACTM), which is crucial for the simulation of particulate flows, providing a more realistic representation of particulate dynamics. This code is designed to simulate multiphase flow and fluid-structure interaction (FSI) problems, capable of running on both CPUs and GPUs with or without subcycling, meaning it can operate efficiently on different hardware platforms, leveraging the parallel processing capabilities of modern computational architectures to accelerate simulation processes. Code-level optimizations, such as loop unrolling (pragma unroll), further enhance computational efficiency. IAMReX can handle complex multiphase flow physical models, including the behavior of ideal gases and non-ideal fluids, and employs models like the Wallis speed of sound to calculate the speed of sound in multiphase mixtures, which is crucial for simulating the propagation of pressure waves and similar issues. As a powerful tool, IAMReX is suitable for researchers and engineers who require high precision and efficiency in solving multiphase flow and fluid-structure interaction problems, and it is adaptable to the ever-changing computational demands and challenges.
[@pyroI] [@castro] [@maestro].

# Acknowledgements

The work at Stony Brook was supported by DOE/Office of Nuclear Physics
grant DE-FG02-87ER40317 and DOE grant DE-SC0017955.

# References
