# IAMReX

This repo builds upon IAMR (https://amrex-codes.github.io/IAMR/) code that solves the multiphase incompressible flows. The Navier-Stokes euqations are solved on a semi-staggered grid using the projection method. The gas-liquid interface is captured using either the level set (LS) method or the convervative level set (CLS) method. The fluid-solid interface is resolved using the diffused immersed boundary method (DIBM). The particle-wall as well as the particle-particle collision are also captured by the adaptive collision
time model (ACTM). This code aims at simulating the multiphase flow and fluid structure interaction (FSI) problems on both CPUs and GPUs with/without subcycling.

## Features

- LS method and reinitialization schemes
- Diffused Immersed Boundary Method
- Particle Collision Algorithms

## Examples

- [Reversed Single Vortex (RSV)](./Tutorials/RSV/)

<div align="center">
    <img src="./README_figures/RSV.jpeg" alt="Profiles of drop interface in the RSV problem" width="500">
    <br>
    <figcaption style="text-align:center;">Profiles of drop interface in the RSV problem at t/T=1 after one rotation. Black line: Analytical Solution; Red line: 64*64; Blue line: 128*128; Green line: 256*256</figcaption>
    <br>
    <br>
</div>

- [Rayleigh-Taylor (RT) instability](./Tutorials/RayleighTaylor_LS/)

<div align="center">
    <!-- First Image -->
    <div style="display:inline-block; margin-right:10px; vertical-align:top; width:600px; height:400px; overflow:hidden;">
        <img src="./README_figures/RT_IAMR.png" alt="Short Description 1" width="600">
        <br>
        <figcaption>Density profile at t/T=2.42 using IAMR convective scheme.</figcaption>
        <br>
        <br>        
    </div>
    <!-- Second Image -->    
    <div style="display:inline-block; margin-left:10px; vertical-align:top; width:600px; height:400px; overflow:hidden;">
        <img src="./README_figures/RT_LSAMR.png" alt="Short Description 2" width="600">
        <br>
        <figcaption>Density profile at t/T=2.42 using LS method.</figcaption>
        <br>
        <br>
    </div>
    <!-- Third Image -->    
    <div style="display:inline-block; margin-left:10px; vertical-align:top; width:600px; height:400px; overflow:hidden;">
        <img src="./README_figures/RT_tip.png" alt="Short Description 3" width="600">
        <br>
        <figcaption>Comparison of the tip locations of the falling fluid and the rising fluid.</figcaption>
        <br>
        <br>
    </div>
</div>

- [Breaking Wave](./Tutorials/BreakingWave_LS/)