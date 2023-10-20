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

<div align="center">
    <img src="./README_figures/RSV.jpeg" alt="Profiles of drop interface in the RSV problem" width="500">
    <br>
    <figcaption style="text-align:center;">Profiles of drop interface in the RSV problem at t/T=1 after one rotation. Black line: Analytical Solution; Red line: 64*64; Blue line: 128*128; Green line: 256*256</figcaption>
</div>

- [Rayleigh-Taylor (RT) instability](./Tutorials/RayleighTaylor_LS/)

<div align="center">
    <table>
        <tr>
            <td>
                <video width="600" height="600" autoplay loop muted>
                    <source src="./README_figures/movie_IAMR.mpg" type="video/mpg">
                    <!-- Your browser does not support the video tag. -->
                </video>
                <br>
                <figcaption>Description for Movie 1</figcaption>
            </td>
            <td>
                <video width="600" height="600" autoplay loop muted>
                    <source src="./README_figures/movie_LSAMR.mpg" type="video/mpg">
                    <!-- Your browser does not support the video tag. -->
                </video>
                <br>
                <figcaption>Description for Movie 2</figcaption>
            </td>
        </tr>
    </table>
</div>

