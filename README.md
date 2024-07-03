# IAMReX

This repo builds upon IAMR (https://amrex-codes.github.io/IAMR/) code that solves the multiphase incompressible flows. The Navier-Stokes euqations are solved on a semi-staggered grid using the projection method. The gas-liquid interface is captured using either the level set (LS) method or the conservative level set (CLS) method. The fluid-solid interface is resolved using the diffused immersed boundary method (DIBM). The particle-wall as well as the particle-particle collisions are also captured by the adaptive collision
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

## Install

### Download

Our code is rely on AMReX framework, you must download such repo as follow: 

1. AMReX:`git clone https://github.com/ruohai0925/amrex`
2. AMReX-Hydro:`git clone https://github.com/ruohai0925/AMReX-Hydro`
3. this repo:`git clone https://github.com/ruohai0925/IAMR/tree/development`

After that, you will find three folder in your current directory:`AMReX`,`AMReX-Hydro`,`IAMR`.

### Compile

We recommend using the GNU compiler to compile the program on the Linux platform. The compilation process requires preparing a `make` file, which you can find in the example folder under Tutorials. It is strongly recommended to use the `GNUmakefile` prepared in advance by the example.

For example, if we want to compile in the `FlowPastSphere` , refer to the following steps:

1. `cd` to the FlowPastSphere directory

    ```shell
    cd IAMR/Tutorials/FlowPastSphere
    ```

2. Modify compilation parameters in `GNUmakefile`.

   The compilation parameters depend on your computing platform. If you use `MPI` to run your program, then set `USE_MPI = TRUE`. If you are running the program with `Nvidia GPU(CUDA)`, then set `USE_CUDA = TRUE`. When using GPU runtime, please make sure your CUDA environment is ok. For the compilation parameters in the file, you can find the relevant information in [options](https://amrex-fluids.github.io/IAMR/Getting_Started.html#building-the-code).
   
3. Compile

   After preparing the above settings, you can compile the program:

   ```shell
   make
   ```

   You can add parameters after `make`, such as `-f your_make_file` to customize your parameter file, and `-j 8` to specify `8` threads to participate in the compilation.

   If the compilation is successful, an executable file will be generated. Usually the executable file is named in the following format:`amr[DIM].[Compiler manufacturers].[computing platform].[is Debug].ex`.The executable file name of the Release version of the three-dimensional compiled using the GNU compiler in the MPI environment is: `amr3d.GNU.MPI.ex`.
   

### Run your codes

you should takes an input file as its first command-line argument.  The file may contain a set of parameter definitions that will overrides defaults set in the code. In the `FlowPastSphere`, you can find a file named `inputs.3d.flow_past_spher`, run:

```shell
./amr3d.GNU.MPI.ex inputs.3d.flow_past_spher
```

If you use `MPI` to run your program, you can type:

```shell
mpirun -np how_many_threads amr3d.GNU.MPI.ex inputs.3d.flow_past_spher
```

this code typically generates subfolders in the current folder that are named `plt00000`, `plt00010`, etc, and `chk00000`, `chk00010`, etc. These are called plotfiles and checkpoint files. The plotfiles are used for visualization of derived fields; the checkpoint files are used for restarting the code. 

​    

​    