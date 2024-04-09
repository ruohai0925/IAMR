#ifndef DIFFUSEDIB_H
#define DIFFUSEDIB_H

#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>
#include <fstream>

using namespace amrex;
// using deltaFuncType = std::function<AMREX_GPU_HOST_DEVICE void(Real, Real, Real, Real&)>;

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

enum DELTA_FUNCTION_TYPE{
    FOUR_POINT_IB = 0,
    THREE_POINT_IB
};

struct kernel{
    RealVect velocity;
    RealVect location;
    RealVect omega;
    RealVect varphi;
    Real radious;
    Real rho;
    int ml;
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
                       const BoxArray & ba)
                       : ParticleContainer<0, 0, numAttri>(gm, dm, ba){}

    void InitParticles(const Vector<Real>& x,
                       const Vector<Real>& y,
                       const Vector<Real>& z,
                       Real rho_s,
                       Real radious,
                       Real rho_f, 
                       int force_index, 
                       int velocity_index,
                       int finest_level);

    void InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce, int loop_time = 20, Real dt = 0.1, Real alpha_k = 0.5, DELTA_FUNCTION_TYPE type = FOUR_POINT_IB);

    void WriteParticleFile(int index);

// private:
    int euler_finest_level;

    int euler_force_index;

    int euler_velocity_index;

    Real euler_fluid_rho;

    Vector<kernel> particle_kernels;

    void InitialWithLargrangianPoints(const kernel& current_kernel);

    void VelocityInterpolation(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

    void ForceSpreading(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

    void UpdateParticles(const amrex::MultiFab& Euler, kernel& kernel, Real dt, Real alpha_k );

    void ComputeLagrangianForce(Real dt, const kernel& kernel);

    int verbose = 1;
};

class Particles{
public:
    static void create_particles(const Geometry &gm,
                                 const DistributionMapping & dm,
                                 const BoxArray & ba);

    static mParticle* get_particles();
    static void init_particle();

    static void define_para(const Vector<Real>& x,
                            const Vector<Real>& y,
                            const Vector<Real>& z,
                            Real rho_s,
                            Real radious,
                            Real rho_f, 
                            int force_index, 
                            int velocity_index,
                            int finest_level);

private:
    inline static Vector<Real> _x{}, _y{}, _z{};
    inline static int euler_finest_level = 0;
    inline static int euler_velocity_index = 0;
    inline static int euler_force_index = 0;
    inline static Real euler_fluid_rho = 0.0;
    inline static Real euler_solid_rho = 0.0;
    inline static Real particle_radious = 0.0;

    inline static mParticle* particle = nullptr;
};

#endif
