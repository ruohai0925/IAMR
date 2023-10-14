
#ifndef NS_KERNELS_H
#define NS_KERNELS_H

#include <AMReX_MultiFab.H>

using namespace amrex;

void cc_to_cc_grad(Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM>& phi_cc_grad, const MultiFab& phi, const Geometry& geom, int normalize);
void cc_grad_to_cc_div(MultiFab& phi_cc_div, Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM>& phi_cc_grad, const Geometry& geom);
void cc_to_cc_lap(MultiFab& phi_cc_lap, MultiFab& phi, const Geometry& geom);
void cc_to_fc(Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM>& phi_fc, const MultiFab& phi, const Geometry& geom);
#endif
