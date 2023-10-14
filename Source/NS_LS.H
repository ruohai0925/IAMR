
#ifndef NS_LS_H
#define NS_LS_H

#include <AMReX_MultiFab.H>

using namespace amrex;

Real calculate_eps_one (const Geometry& geom, int reinit_levelset);
Real calculate_eps_two (const Geometry& geom, int reinit_levelset);
void levelset_diffcomp (Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM>& phi_cc_grad,
                        MultiFab& phi_ctime,
                        MultiFab& phi1,
                        MultiFab& phi2,
                        int epsG1,
                        int epsG2);

#endif
