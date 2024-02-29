#ifndef DIFFUSEDIB_H
#define DIFFUSEDIB_H

#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <eigen3/Eigen/Dense>

using namespace amrex;
using mVector = Eigen::Vector<Real, 3>;
using deltaFuncType = std::function<void(Real, Real, Real, Real&)>;

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
struct kernel{
    mVector velocity;
    mVector location;
    mVector omega;
    mVector varphi;
    Real radious;
    Real rho;
    Real ml;
    Real dv;
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
void deltaFunction(Real xf, Real xp, Real h, Real& value);

class mParticle : public ParticleContainer<0, 0, numAttri>{
public:

    explicit mParticle(const Vector<Geometry>            & gm,
                       const Vector<DistributionMapping> & dm,
                       const Vector<BoxArray>            & ba,
                       const Vector<int>                 & rr)
                       : ParticleContainer<0, 0, numAttri>(gm, dm, ba, rr){}

    explicit mParticle(const Geometry &gm,
                       const DistributionMapping & dm,
                       const BoxArray & ba,
                       Real fluid_rho, 
                       int force_index, 
                       int velocity_index, 
                       int finest_level)
                       : ParticleContainer<0, 0, numAttri>(gm, dm, ba),
                       euler_fluid_rho(fluid_rho),
                       euler_finest_level(finest_level),
                       euler_force_index(force_index),
                       euler_velocity_index(force_index){}

    void InitParticlesAndMarkers(const Vector<Real>& x,
                                 const Vector<Real>& y,
                                 const Vector<Real>& z,
                                 Real rho_s,
                                 int radious);

    void InteractWithEuler(MultiFab &Euler, int loop_time = 20, Real dt = 0.1, Real alpha_k = 0.5, const deltaFuncType& delta = deltaFunction);

    void WriteParticleFile(int index);

private:
    int euler_finest_level;

    int euler_force_index;

    int euler_velocity_index;

    Real euler_fluid_rho;

    Vector<kernel> particle_kernels;

    void InitialWithLargrangianPoints(const kernel& current_kernel);

    void VelocityInterpolation(const amrex::MultiFab &Euler, const deltaFuncType& delta);

    void ForceSpreading(amrex::MultiFab &Euler, const deltaFuncType& delta);

    void VelocityCorrection(const amrex::MultiFab& Euler, kernel& kernel, Real dt, Real alpha_k );

    void UpdateF(Real dt, const kernel& kernel);
};

#endif
