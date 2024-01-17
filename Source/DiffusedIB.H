
#ifndef DIFFUSEDIB_H
#define DIFFUSEDIB_H

#include <AMReX_MultiFab.H>

using namespace amrex;

AMREX_INLINE AMREX_GPU_DEVICE
Real nodal_phi_to_heavi(Real phi)
{
    if (phi <= 0.0) {
        return 0.0;
    }
    else {
        return 1.0;
    }
}

void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal);

#endif
