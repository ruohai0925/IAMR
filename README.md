# IAMR-MULTIPHASE

This repo builds upon IAMR (https://amrex-codes.github.io/IAMR/) code that solves the two-phase 
incompressible Navier-Stokes equations using the level set (LS) method. This code aims at simulating the multiphase flow and fluid structure interaction (FSI)
problems on CPUs and GPUs with/without subcycling.

## Features

- LS method and reinitialization schemes

## Tutorials

- [Reversed Single Vortex (RSV)](./Tutorials/RSV/)

<!-- <div align="center">
<img src="./README_figures/RSV.jpeg" alt="Profiles of drop interface in the RSV problem at t/T=1 after one rotation. Black line: Analytical Solution; Red line: 64*64; Blue line: 128*128; Green line: 256*256"> -->

<img src="./README_figures/RSV.jpeg" alt="Profiles of drop interface in the RSV problem" width="500">
<figcaption>Profiles of drop interface in the RSV problem at t/T=1 after one rotation. Black line: Analytical Solution; Red line: 64*64; Blue line: 128*128; Green line: 256*256</figcaption>

- Rayleigh-Taylor (RT) instability

