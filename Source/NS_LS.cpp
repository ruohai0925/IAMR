
#include <NS_LS.H>

#include <AMReX_ParmParse.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_FillPatchUtil.H>
#include <iamr_constants.H>

using namespace amrex;

// ls related
Real calculate_eps_one (const Geometry& geom, int reinit_cons_levelset)
{

    Print() << "In the calculate_eps_one " << std::endl;

    Real eps_one = 0.0;

    const Real* dx    = geom.CellSize();
    Real dxmax        = dx[0];
    for (int d=1; d<AMREX_SPACEDIM; ++d) {
        dxmax = std::max(dxmax,dx[d]);
    }

    if (reinit_cons_levelset == 1)
    {
        // sussman
    }
    else if (reinit_cons_levelset == 2)
    {
        // improved conservative level set, JCP, 2007
    }
    else if (reinit_cons_levelset == 3)
    {
        // multiUQ
        eps_one = 1.125 * dxmax;
    }

    return eps_one;
}

Real calculate_eps_two (const Geometry& geom, int reinit_cons_levelset)
{

    Print() << "In the calculate_eps_two " << std::endl;

    Real eps_two = 0.0;

    const Real* dx    = geom.CellSize();
    Real dxmax        = dx[0];
    for (int d=1; d<AMREX_SPACEDIM; ++d) {
        dxmax = std::max(dxmax,dx[d]);
    }

    if (reinit_cons_levelset == 1)
    {
        // sussman
    }
    else if (reinit_cons_levelset == 2)
    {
        // improved conservative level set, JCP, 2007
    }
    else if (reinit_cons_levelset == 3)
    {
        // multiUQ
        eps_two = 0.125 * dxmax;
    }

    return eps_two;
}

