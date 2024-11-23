---
title: 'IAMReX: an adaptive framework for the multiphase flow and fluid-particle interaction problems'
tags:
  - C++
  - Computational Fluid Dynamics
  - Adaptive Mesh Refinement
  - Immersed Boundary Method
  - Multiphase Flow
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

IAMReX is an adaptive C++ simulation framework designed for multiphase flow and fluid-particle interaction problems. It is built in a objected-oriented style, allowing for testing and simulating large-scale problems (e.g., gas-fluid interaction, cluster of particles) in parallel.

The original goal of IAMReX is to extend the capability of IAMR code [@almgren1998conservative], which only uses a density-based solver to capture the diffused interface of the two-phase flow.  IAMReX offers the Level Set (LS) method and the reinitialization techniques for accurately capturing the two-phase interface [@zeng2022parallel], which increases the robustness of simulations with high Reynolds number [@zeng2023consistent]. For fluid-particle interaction problems, IAMReX employs the multidirect forcing immersed boundary method [@li2024open]. The associated Lagrangian markers used to resolve fluid-particle interface only exist on the finest-level grid, which greatly reduces memory usage. Both the subcycling and non-subcycling time advancement methods are inplemented, and these methods help to decouple the time advancement at different levels. In addition, IAMReX is a publicly accessible platform designed specifically for developing massively parallel block-structured adaptive mesh refinement (BSAMR) applications. The code now supports hybrid parallelization using either pure MPI or MPI+OpenMP for multicore machines with the help of the AMReX framework [@zhang2019amrex].

The IAMReX code has undergone considerable development since 2023 and gained a few new contributors in the past two years. Although the projection-based flow solver is inherited from IAMR, IAMReX has added over 3,000 lines of new code, introduced 10 more new test cases, and contributed approximately 60 new commits on GitHub. The versatility, accuracy, and efficiency of the present IAMReX framework are demonstrated by simulating two-phase flow and fluid-particle interaction problems with various types of kinematic constraints. We carefully designed the document such that users can easily compile and run cases. Input files, profiling scripts, and raw postprocessing data are also available for reproducing all results [@liu2024investigate].

# Acknowledgements

C.L., X.L., Y.Z., and Z.Z. are grateful to Ann Almgren, Andy Nonaka, Andrew Myers, Axel Huebl, and Weiqun Zhang in the Lawrence Berkeley National Laboratory (LBNL) for their discussions related to AMReX and IAMR. Y.Z. and Z.Z. also thank Prof. Lian Shen, Prof.~Ruifeng Hu, and Prof. Xiaojing Zheng during their Ph.D. studies.

# References
