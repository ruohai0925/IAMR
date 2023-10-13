
#ifndef NS_LS_H
#define NS_LS_H

#include <AMReX_MultiFab.H>

using namespace amrex;

Real calculate_eps_one (const Geometry& geom, int reinit_cons_levelset);
Real calculate_eps_two (const Geometry& geom, int reinit_cons_levelset);

#endif
