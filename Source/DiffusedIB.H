#ifndef DIFFUSEDIB_H
#define DIFFUSEDIB_H

#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                     particle and markers                      */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

enum P_ATTR{
    U_Marker = 0,
    V_Marker,
    W_Marker,
    Fx_Marker,
    Fy_Marker,
    Fz_Marker,
    numAttri
};

class mParIter : public ParIter<0, 0, numAttri>{
public:
    using amrex::ParIter<0, 0, numAttri>::ParIter;
    using RealVector = amrex::ParIter<0, 0, numAttri>::ContainerType::RealVector;
 
    [[nodiscard]] const std::array<RealVector, numAttri>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    [[nodiscard]] const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    std::array<RealVector, numAttri>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

/**
 * @brief delta function (default)
 * 
 * @param xf 
 * @param xp 
 * @param h 
 * @param value 
 */
void deltaFunction(int xf, Real h, Real& value);

class mParticle : public ParticleContainer<0, 0, numAttri>{
public:
    explicit mParticle(const Vector<Geometry>            & gm,
                       const Vector<DistributionMapping> & dm,
                       const Vector<BoxArray>            & ba,
                       const Vector<int>                 & rr)
                       : ParticleContainer<0, 0, numAttri>(gm, dm, ba, rr){}

    explicit mParticle(const Geometry &gm,
                       const DistributionMapping & dm,
                       const BoxArray & ba)
                       : ParticleContainer<0, 0, numAttri>(gm, dm, ba){}

    void InitParticlesAndMarkers(const Vector<Real>& x,
                                 const Vector<Real>& y,
                                 const Vector<Real>& z,
                                 int radious);

    void VelocityInterpolation(const amrex::MultiFab &Eular, const std::function<void(int, Real, Real&)>& delta = deltaFunction);

    void ForceSpreading(amrex::MultiFab &Eular, const std::function<void(int, Real, Real&)>& delta = deltaFunction);

    void VelocityCorrection(amrex::MultiFab &Eular, Real dt, Real rhoS);

private:
    Vector<int> numMakerPerP;
};

#endif
