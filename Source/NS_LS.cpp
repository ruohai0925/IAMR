
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
Real calculate_eps_one (const Geometry& geom, int reinit_levelset)
{

    Print() << "In the calculate_eps_one " << std::endl;

    Real eps_one = 0.0;

    const Real* dx    = geom.CellSize();
    Real dxmax        = dx[0];
    for (int d=1; d<AMREX_SPACEDIM; ++d) {
        dxmax = std::max(dxmax,dx[d]);
    }

    if (reinit_levelset == 2)
    {
        // improved conservative level set, JCP, 2007
        eps_one = 0.5 * std::pow(dxmax, 0.9);
    }
    else if (reinit_levelset == 3)
    {
        // multiUQ, JCP, 2019/CPC, 2021
        eps_one = 0.875 * dxmax;
    }

    return eps_one;
}

Real calculate_eps_two (const Geometry& geom, int reinit_levelset)
{

    Print() << "In the calculate_eps_two " << std::endl;

    Real eps_two = 0.0;

    const Real* dx    = geom.CellSize();
    Real dxmax        = dx[0];
    for (int d=1; d<AMREX_SPACEDIM; ++d) {
        dxmax = std::max(dxmax,dx[d]);
    }

    if (reinit_levelset == 2)
    {
        // improved conservative level set, JCP, 2007
        eps_two = 0.0;
    }
    else if (reinit_levelset == 3)
    {
        // multiUQ, JCP, 2019/CPC, 2021
        eps_two = 0.125 * dxmax;
    }

    return eps_two;
}

void levelset_diffcomp (Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM>& phi_cc_grad,
                        MultiFab& phi_ctime,
                        MultiFab& phi1,
                        MultiFab& phi2, 
                        int epsG,
                        int epsG2)
{

    Print() << "In the levelset_diffcomp " << std::endl;

    // Step 1: calculate the comp term phi1

    // Step 2: calculate the diff terms phi2


}
