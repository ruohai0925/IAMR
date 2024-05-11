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
    P_thetaK,
    P_phiK,
    numAttri
};

enum DELTA_FUNCTION_TYPE{
    FOUR_POINT_IB = 0,
    THREE_POINT_IB
};

struct Plane{
    IntVect normal{0, 0, 0};
    Real d;
};

struct kernel{
    int id;
    RealVect velocity{0.0,0.0,0.0};
    RealVect location{0.0,0.0,0.0};
    RealVect omega{0.0,0.0,0.0};

    RealVect velocity_old{0.0,0.0,0.0};
    RealVect location_old{0.0,0.0,0.0};
    RealVect omega_old{0.0,0.0,0.0};

    RealVect varphi{0.0,0.0,0.0};
    Real radius{0.0};
    Real rho{0.0};
    int ml{0};
    Real dv{0.0};
    RealVect ib_force{0.0,0.0,0.0};
    RealVect ib_moment{0.0, 0.0, 0.0};

    RealVect sum_u_new{0.0,0.0,0.0};
    RealVect sum_u_old{0.0,0.0,0.0};
    RealVect sum_t_new{0.0,0.0,0.0};
    RealVect sum_t_old{0.0,0.0,0.0};

    RealVect Fcp{0.0,0.0,0.0};
    RealVect Tcp{0.0,0.0,0.0};

    IntVect TL{0, 0, 0}, RL{0, 0, 0};
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
                       const Vector<Real>& rho_s,
                       const Vector<Real>& Vx,
                       const Vector<Real>& Vy,
                       const Vector<Real>& Vz,
                       const Vector<Real>& Ox,
                       const Vector<Real>& Oy,
                       const Vector<Real>& Oz,
                       const Vector<int>& TLX,
                       const Vector<int>& TLY,
                       const Vector<int>& TLZ,
                       const Vector<int>& RLX,
                       const Vector<int>& RLY,
                       const Vector<int>& RLZ,
                       Real radius,
                       Real gravity,
                       int verbose = 0);

    void InteractWithEuler(int iStep, amrex::Real time, MultiFab &EulerVel, MultiFab &EulerForce, Real dt = 0.1, DELTA_FUNCTION_TYPE type = FOUR_POINT_IB);

    void WriteParticleFile(int index);

    Vector<kernel> particle_kernels;

    Vector<Plane> boundary_plane;

    void InitialWithLargrangianPoints(const kernel& current_kernel);

    void ResetLargrangianPoints(Real dt);

    void VelocityInterpolation(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

    void ComputeLagrangianForce(Real dt, const kernel& kernel);

    void ForceSpreading(amrex::MultiFab &Euler, kernel& kernel, DELTA_FUNCTION_TYPE type);

    void VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const;

    void UpdateParticles(const MultiFab& Euler_old, const MultiFab& Euler, MultiFab& phi_nodal, MultiFab& pvf, Real dt);

    static void WriteIBForceAndMoment(int step, amrex::Real time, kernel& current_kernel);

    void RecordOldValue(kernel& kernel);

    uint32_t ib_force_file_index = 0;

    RealVect m_gravity{0.0,0.0,0.0};

    int verbose = 0;
};

class Particles{
public:
    static void create_particles(const Geometry &gm,
                                 const DistributionMapping & dm,
                                 const BoxArray & ba);

    static mParticle* get_particles();
    static void init_particle( int level, Real gravity);
    static void Initialize();
    static int ParticleFinestLevel();
    static void define_para(const Vector<Real>& x,
                            const Vector<Real>& y,
                            const Vector<Real>& z,
                            Real rho_s,
                            Real radius,
                            Real rho_f, 
                            int force_index, 
                            int velocity_index,
                            int finest_level);

private:
    inline static mParticle* particle = nullptr;
};

#endif
