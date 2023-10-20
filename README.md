# IAMR-MULTIPHASE

This repo builds upon IAMR (https://amrex-codes.github.io/IAMR/) code that solves the two-phase 
incompressible Navier-Stokes equations using the level set (LS) method. This code aims at simulating the multiphase flow and fluid structure interaction (FSI)
problems on CPUs and GPUs with/without subcycling.

## Features

- LS method and reinitialization schemes

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
        <figcaption>Density profile at t/T=1.1 using IAMR convective scheme.</figcaption>
        <br>
        <br>        
    </div>
    <!-- Second Image -->    
    <div style="display:inline-block; margin-left:10px; vertical-align:top; width:600px; height:400px; overflow:hidden;">
        <img src="./README_figures/RT_LSAMR.png" alt="Short Description 2" width="600">
        <br>
        <figcaption>Density profile at t/T=1.1 using LS method.</figcaption>
        <br>
        <br>
    </div>
</div>

- [Breaking Wave](./Tutorials/BreakingWave_LS/)