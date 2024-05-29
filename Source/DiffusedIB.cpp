
#include <AMReX_Math.H>
#include <AMReX_Print.H>
#include <DiffusedIB.H>

#include <AMReX_ParmParse.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_FillPatchUtil.H>
#include <iamr_constants.H>

#include <filesystem>
namespace fs = std::filesystem;

#define GHOST_CELLS 2

using namespace amrex;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                     global variable                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define LOCAL_LEVEL 0

const Vector<std::string> direction_str{"X","Y","Z"};

namespace ParticleProperties{
    Vector<Real> _x{}, _y{}, _z{}, _rho{};
    Vector<Real> Vx{}, Vy{}, Vz{};
    Vector<Real> Ox{}, Oy{}, Oz{};
    Vector<Real> _radius;
    Vector<int> TLX{}, TLY{},TLZ{},RLX{},RLY{},RLZ{};
    int euler_finest_level{0};
    int euler_velocity_index{0};
    int euler_force_index{0};
    Real euler_fluid_rho{0.0};
    int verbose{0};
    int loop_ns{2};
    int loop_solid{1};
    int Uhlmann{0};

    Vector<Real> GLO, GHI;
    int start_step{-1};
    int collision_model{0};

    GpuArray<Real, 3> plo{0.0,0.0,0.0}, phi{0.0,0.0,0.0}, dx{0.0, 0.0, 0.0};
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                     other function                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal)
{

    amrex::Print() << "In the nodal_phi_to_pvf\n";

    pvf.setVal(0.0);

    // Only set the valid cells of pvf
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(pvf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& pvffab   = pvf.array(mfi);
        auto const& pnfab = phi_nodal.array(mfi);
        amrex::ParallelFor(bx, [pvffab, pnfab]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            Real num = 0.0;
            for(int kk=k; kk<=k+1; kk++) {
                for(int jj=j; jj<=j+1; jj++) {
                    for(int ii=i; ii<=i+1; ii++) {
                        num += (-pnfab(ii,jj,kk)) * nodal_phi_to_heavi(-pnfab(ii,jj,kk));
                    }
                }
            }
            Real deo = 0.0;
            for(int kk=k; kk<=k+1; kk++) {
                for(int jj=j; jj<=j+1; jj++) {
                    for(int ii=i; ii<=i+1; ii++) {
                        deo += std::abs(pnfab(ii,jj,kk));
                    }
                }
            }
            pvffab(i,j,k) = num / (deo + 1.e-12);
        });
    }

}

void calculate_phi_nodal(MultiFab& phi_nodal, kernel& current_kernel)
{
    phi_nodal.setVal(0.0);

    amrex::Real Xp = current_kernel.location[0];
    amrex::Real Yp = current_kernel.location[1];
    amrex::Real Zp = current_kernel.location[2];
    amrex::Real Rp = current_kernel.radius;

    // Only set the valid cells of phi_nodal
    for (MFIter mfi(phi_nodal,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& pnfab = phi_nodal.array(mfi);
        const auto *pnfab_ptr = &pnfab;
        auto *dx = &ParticleProperties::dx;
        auto *plo = &ParticleProperties::plo;
        amrex::ParallelFor(bx, [=]
            AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Real Xn = i * (*dx)[0] + (*plo)[0];
                Real Yn = j * (*dx)[1] + (*plo)[1];
                Real Zn = k * (*dx)[2] + (*plo)[2];

                (*pnfab_ptr)(i,j,k) = std::sqrt( (Xn - Xp)*(Xn - Xp)
                        + (Yn - Yp)*(Yn - Yp)  + (Zn - Zp)*(Zn - Zp)) - Rp;
                (*pnfab_ptr)(i,j,k) = (*pnfab_ptr)(i,j,k) / Rp;

            }
        );
    }
}

// May use ParReduce later, https://amrex-codes.github.io/amrex/docs_html/GPU.html#multifab-reductions
void CalculateSumU_cir (RealVect& sum,
                        const MultiFab& E,
                        const MultiFab& pvf,
                        int EulerVelIndex)
{
    const Real d = Math::powi<3>(ParticleProperties::dx[0]);

    for (MFIter mfi(E,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& uarray = E.array(mfi);
        auto const& pvffab = pvf.array(mfi);
        const auto * new_ptr = &uarray;
        const auto * pvf_ptr = &pvffab;
        auto* sum_ptr = &sum;
        amrex::ParallelFor(bx, [=]
            AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Gpu::Atomic::AddNoRet(&(*sum_ptr)[0], (*new_ptr)(i, j, k, EulerVelIndex    ) * d * (*pvf_ptr)(i, j, k));
                Gpu::Atomic::AddNoRet(&(*sum_ptr)[1], (*new_ptr)(i, j, k, EulerVelIndex + 1) * d * (*pvf_ptr)(i, j, k));
                Gpu::Atomic::AddNoRet(&(*sum_ptr)[2], (*new_ptr)(i, j, k, EulerVelIndex + 2) * d * (*pvf_ptr)(i, j, k));
            }
        );
    }
}

void CalculateSumT_cir (RealVect& sum,
                        const MultiFab& E,
                        const MultiFab& pvf,
                        const RealVect pLoc,
                        int EulerVelIndex)
{
    const Real d = Math::powi<3>(ParticleProperties::dx[0]);
    auto plo = ParticleProperties::plo;
    auto dx = ParticleProperties::dx;
    for (MFIter mfi(E,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& uarray = E.array(mfi);
        auto const& pvffab = pvf.array(mfi);
        const auto * new_ptr = &uarray;
        const auto * pvf_ptr = &pvffab;
        auto* sum_ptr = &sum;
        amrex::ParallelFor(bx, [=]
            AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Real x = plo[0] + i*dx[0] + 0.5*dx[0];
                Real y = plo[1] + i*dx[1] + 0.5*dx[1];
                Real z = plo[2] + i*dx[2] + 0.5*dx[2];
                Real vx = (*new_ptr)(i, j, k, EulerVelIndex    );
                Real vy = (*new_ptr)(i, j, k, EulerVelIndex + 1);
                Real vz = (*new_ptr)(i, j, k, EulerVelIndex + 2);
                RealVect tmp = RealVect(x - pLoc[0], y - pLoc[1], z - pLoc[2]).crossProduct(RealVect(vx, vy, vz));
                Gpu::Atomic::AddNoRet(&(*sum_ptr)[0], tmp[0] * d * (*pvf_ptr)(i, j, k));
                Gpu::Atomic::AddNoRet(&(*sum_ptr)[1], tmp[1] * d * (*pvf_ptr)(i, j, k));
                Gpu::Atomic::AddNoRet(&(*sum_ptr)[2], tmp[2] * d * (*pvf_ptr)(i, j, k));
            }
        );
    }

}

[[nodiscard]] AMREX_FORCE_INLINE
Real cal_momentum(Real rho, Real radius)
{
    return 8.0 * Math::pi<Real>() * rho * Math::powi<5>(radius) / 15.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void deltaFunction(Real xf, Real xp, Real h, Real& value, DELTA_FUNCTION_TYPE type)
{
    Real rr = amrex::Math::abs(( xf - xp ) / h);

    switch (type) {
    case DELTA_FUNCTION_TYPE::FOUR_POINT_IB:
        if(rr >= 0 && rr < 1 ){
            value = 1.0 / 8.0 * ( 3.0 - 2.0 * rr + std::sqrt( 1.0 + 4.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else if (rr >= 1 && rr < 2) {
            value = 1.0 / 8.0 * ( 5.0 - 2.0 * rr - std::sqrt( -7.0 + 12.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    case DELTA_FUNCTION_TYPE::THREE_POINT_IB:
        if(rr >= 0.5 && rr < 1.5){
            value = 1.0 / 6.0 * ( 5.0 - 3.0 * rr - std::sqrt( - 3.0 * Math::powi<2>( 1 - rr) + 1.0 )) / h;
        }else if (rr >= 0 && rr < 0.5) {
            value = 1.0 / 3.0 * ( 1.0 + std::sqrt( 1.0 - 3 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    default:
        break;
    }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                    mParticle member function                  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
//loop all particels
void mParticle::InteractWithEuler(int iStep, 
                                  amrex::Real time, 
                                  MultiFab &EulerVel, 
                                  MultiFab &EulerForce, 
                                  Real dt,
                                  DELTA_FUNCTION_TYPE type)
{
    if (verbose) amrex::Print() << "[Particle] mParticle::InteractWithEuler\n";

    for(kernel& kernel : particle_kernels){
        InitialWithLargrangianPoints(kernel); // Initialize markers for a specific particle
        ResetLargrangianPoints(dt);
        
        amrex::Real ib_force_x  = 0.0;
        amrex::Real ib_force_y  = 0.0;
        amrex::Real ib_force_z  = 0.0;
        amrex::Real ib_moment_x = 0.0;
        amrex::Real ib_moment_y = 0.0;
        amrex::Real ib_moment_z = 0.0;
        
        //for 1 -> Ns
        int loop = ParticleProperties::loop_ns;
        BL_ASSERT(loop > 0);
        while(loop > 0){
            if(verbose) amrex::Print() << "[Particle] Ns loop index : " << loop << "\n";
            
            VelocityInterpolation(EulerVel, type);
            ComputeLagrangianForce(dt, kernel);
            
            EulerForce.setVal(0.0, ParticleProperties::euler_force_index, 3, GHOST_CELLS); // clear Euler force
            kernel.ib_force.scale(0.0); // clear kernel ib_force
            kernel.ib_moment.scale(0.0); // clear kernel ib_moment
            ForceSpreading(EulerForce, kernel, type);

            amrex::ParallelAllReduce::Sum(&kernel.ib_force[0], 3, ParallelDescriptor::Communicator());
            amrex::ParallelAllReduce::Sum(&kernel.ib_moment[0], 3, ParallelDescriptor::Communicator());

            ib_force_x  += kernel.ib_force[0];
            ib_force_y  += kernel.ib_force[1];
            ib_force_z  += kernel.ib_force[2];            
            ib_moment_x += kernel.ib_moment[0];
            ib_moment_y += kernel.ib_moment[1];
            ib_moment_z += kernel.ib_moment[2];  

            VelocityCorrection(EulerVel, EulerForce, dt);
            
            loop--;
        }
        
        kernel.ib_force[0]  = ib_force_x;
        kernel.ib_force[1]  = ib_force_y;
        kernel.ib_force[2]  = ib_force_z;
        kernel.ib_moment[0] = ib_moment_x;
        kernel.ib_moment[1] = ib_moment_y;
        kernel.ib_moment[2] = ib_moment_z;
  
        WriteIBForceAndMoment(iStep, time, kernel);
    }
}

void mParticle::InitParticles(const Vector<Real>& x,
                              const Vector<Real>& y,
                              const Vector<Real>& z,
                              const Vector<Real>& rho_s,
                              const Vector<Real>& Vx,
                              const Vector<Real>& Vy,
                              const Vector<Real>& Vz,
                              const Vector<Real>& Ox,
                              const Vector<Real>& Oy,
                              const Vector<Real>& Oz,
                              const Vector<int>& TLXt,
                              const Vector<int>& TLYt,
                              const Vector<int>& TLZt,
                              const Vector<int>& RLXt,
                              const Vector<int>& RLYt,
                              const Vector<int>& RLZt,
                              const Vector<Real>& radius,
                              Real h,
                              Real gravity,
                              int _verbose)
{
    verbose = _verbose;
    if (verbose) amrex::Print() << "[Particle] mParticle::InitParticles\n";

    m_gravity[2] = gravity;

    //pre judge
    if(!((x.size() == y.size()) && (x.size() == z.size()))){
        Print() << "particle's position container are all different size";
        return;
    }

    //all the particles have different radius
    for(int index = 0; index < x.size(); index++){
        kernel mKernel;
        mKernel.id = index + 1;
        mKernel.location[0] = x[index];
        mKernel.location[1] = y[index];
        mKernel.location[2] = z[index];
        mKernel.velocity[0] = Vx[index];
        mKernel.velocity[1] = Vy[index];
        mKernel.velocity[2] = Vz[index];
        mKernel.omega[0] = Ox[index];
        mKernel.omega[1] = Oy[index];
        mKernel.omega[2] = Oz[index];

        // use current state to initialize old state
        mKernel.location_old = mKernel.location;
        mKernel.velocity_old = mKernel.velocity;
        mKernel.omega_old = mKernel.omega;

        mKernel.TL[0] = TLXt[index];
        mKernel.TL[1] = TLYt[index];
        mKernel.TL[2] = TLZt[index];
        mKernel.RL[0] = RLXt[index];
        mKernel.RL[1] = RLYt[index];
        mKernel.RL[2] = RLZt[index];
        mKernel.rho = rho_s[index];
        mKernel.radius = radius[index];
        mKernel.Vp = Math::pi<Real>() * 4 / 3 * Math::powi<3>(radius[index]);

        int Ml = static_cast<int>( Math::pi<Real>() / 3 * (12 * Math::powi<2>(mKernel.radius / h)));
        Real dv = Math::pi<Real>() * h / 3 / Ml * (12 * mKernel.radius * mKernel.radius + h * h);
        mKernel.ml = Ml;
        mKernel.dv = dv;
        if( Ml > max_largrangian_num ) max_largrangian_num = Ml;

        mKernel.phiK = (Real *)malloc(Ml * sizeof(Real));
        mKernel.thetaK = (Real *)malloc(Ml * sizeof(Real));

        Real phiK = 0;
        for(int marker_index = 0; marker_index < Ml; marker_index++){
            Real Hk = -1.0 + 2.0 * (marker_index) / ( Ml - 1.0);
            Real thetaK = std::acos(Hk);    
            if(marker_index == 0 || marker_index == (Ml - 1)){
                phiK = 0;
            }else {
                phiK = std::fmod( phiK + 3.809 / std::sqrt(Ml) / std::sqrt( 1 - Math::powi<2>(Hk)) , 2 * Math::pi<Real>());
            }
            mKernel.phiK[marker_index] = phiK;
            mKernel.thetaK[marker_index] = thetaK;
        }

        particle_kernels.emplace_back(mKernel);

        if (verbose) amrex::Print() << "h: " << h << ", Ml: " << Ml << ", D: " << Math::powi<3>(h) << "gravity : " << gravity << "\n"
                                    << "Kernel : " << index << ": Location (" << x[index] << ", " << y[index] << ", " << z[index] 
                                    << "), Velocity : (" << mKernel.velocity[0] << ", " << mKernel.velocity[1] << ", "<< mKernel.velocity[2] 
                                    << "), Radius: " << mKernel.radius << ", Ml: " << Ml << ", dv: " << dv << ", Rho: " << mKernel.rho << "\n";
    }
    //collision box generate
    m_Collision.SetGeometry(RealVect(ParticleProperties::GLO), RealVect(ParticleProperties::GHI),particle_kernels[0].radius, h);
}

void mParticle::InitialWithLargrangianPoints(const kernel& current_kernel){

    if (verbose) amrex::Print() << "mParticle::InitialWithLargrangianPoints\n";
    for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        if(np == 0) continue;
        auto *particles = pti.GetArrayOfStructs().data();

        const auto location = current_kernel.location;
        const auto radius = current_kernel.radius;
        auto* phiK = current_kernel.phiK;
        auto* thetaK = current_kernel.thetaK;

        amrex::ParallelFor( np, [=]
            AMREX_GPU_DEVICE (int i) noexcept {
                auto id = particles[i].id();
                particles[i].pos(0) = location[0] + radius * std::sin(thetaK[id - 1]) * std::cos(phiK[id - 1]);
                particles[i].pos(1) = location[1] + radius * std::sin(thetaK[id - 1]) * std::sin(phiK[id - 1]);
                particles[i].pos(2) = location[2] + radius * std::cos(thetaK[id - 1]);
            }
        );
    }
    // Redistribute the markers after updating their locations
    mContainer->Redistribute();
    amrex::Print() << "[particle] : particle num :" << mContainer->TotalNumberOfParticles() << "\n";
    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("particle", 1));
}

template <typename P = Particle<numAttri>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void VelocityInterpolation_cir(int p_iter, P const& p, Real& Up, Real& Vp, Real& Wp,
                               Array4<Real const> const& E, int EulerVIndex,
                               const int *lo, const int *hi, 
                               GpuArray<Real, AMREX_SPACEDIM> const& plo,
                               GpuArray<Real, AMREX_SPACEDIM> const& dx,
                               DELTA_FUNCTION_TYPE type)
{

    //std::cout << "lo " << lo[0] << " " << lo[1] << " "<< lo[2] << "\n";
    //std::cout << "hi " << hi[0] << " " << hi[1] << " "<< hi[2] << "\n";

    const Real d = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

    const Real lx = (p.pos(0) - plo[0]) / dx[0]; // x
    const Real ly = (p.pos(1) - plo[1]) / dx[1]; // y
    const Real lz = (p.pos(2) - plo[2]) / dx[2]; // z

    int i = static_cast<int>(Math::floor(lx)); // i
    int j = static_cast<int>(Math::floor(ly)); // j
    int k = static_cast<int>(Math::floor(lz)); // k

    //std::cout << "p_iter " << p_iter << " p.pos(0): " << p.pos(0) << " p.pos(1): " << p.pos(1) << " p.pos(2): " << p.pos(2) << "\n";

    // std::cout << "d: " << d << "\n"
    //         << "lx: " << lx << ", ly: " << ly << ", lz: " << lz << "\n"
    //         << "i: " << i << ", j: " << j << ", k: " << k << std::endl;

    Up = 0;
    Vp = 0;
    Wp = 0;
    //Euler to largrangian
    for(int ii = -2; ii < 3; ii++){
        for(int jj = -2; jj < 3; jj++){
            for(int kk = -2; kk < 3; kk ++){
                Real tU, tV, tW;
                const Real xi = (i + ii) * dx[0] + dx[0]/2;
                const Real yj = (j + jj) * dx[1] + dx[1]/2;
                const Real kz = (k + kk) * dx[2] + dx[2]/2;
                deltaFunction( p.pos(0), xi, dx[0], tU, type);
                deltaFunction( p.pos(1), yj, dx[1], tV, type);
                deltaFunction( p.pos(2), kz, dx[2], tW, type);
                const Real delta_value = tU * tV * tW;
                Up += delta_value * E(i + ii, j + jj, k + kk, EulerVIndex    ) * d;
                Vp += delta_value * E(i + ii, j + jj, k + kk, EulerVIndex + 1) * d;
                Wp += delta_value * E(i + ii, j + jj, k + kk, EulerVIndex + 2) * d;
            }
        }
    }
}

void mParticle::VelocityInterpolation(MultiFab &EulerVel,
                                      DELTA_FUNCTION_TYPE type)//
{
    if (verbose) amrex::Print() << "\tmParticle::VelocityInterpolation\n";

    //amrex::Print() << "euler_finest_level " << euler_finest_level << std::endl;
    const auto& gm = mContainer->GetParGDB()->Geom(LOCAL_LEVEL);
    auto plo = gm.ProbLoArray();
    auto dx = gm.CellSizeArray();
    // attention
    // velocity ghost cells will be up-to-date
    EulerVel.FillBoundary(ParticleProperties::euler_velocity_index, 3, gm.periodicity());

    const int EulerVelocityIndex = ParticleProperties::euler_velocity_index;

    for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        
        const Box& box = pti.validbox();
        
        auto& particles = pti.GetArrayOfStructs();
        auto *p_ptr = particles.data();
        const Long np = pti.numParticles();

        auto& attri = pti.GetAttribs();
        auto* Up = attri[P_ATTR::U_Marker].data();
        auto* Vp = attri[P_ATTR::V_Marker].data();
        auto* Wp = attri[P_ATTR::W_Marker].data();
        const auto& E = EulerVel.array(pti);

        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            VelocityInterpolation_cir(i, p_ptr[i], Up[i], Vp[i], Wp[i], E, EulerVelocityIndex, box.loVect(), box.hiVect(), plo, dx, type);
        });
    }
    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("particle", 2));
    //amrex::Abort("stop here!");
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ForceSpreading_cic (P const& p,
                         Real Px,
                         Real Py,
                         Real Pz,
                         ParticleReal fxP,
                         ParticleReal fyP,
                         ParticleReal fzP,
                         Real & ib_fx,
                         Real & ib_fy,
                         Real & ib_fz,
                         Real & ib_mx,
                         Real & ib_my,
                         Real & ib_mz,
                         Array4<Real> const& E,
                         int EulerForceIndex,
                         Real dv,
                         GpuArray<Real,AMREX_SPACEDIM> const& plo,
                         GpuArray<Real,AMREX_SPACEDIM> const& dx,
                         DELTA_FUNCTION_TYPE type)
{
    //const Real d = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    //plo to ii jj kk
    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    Real lz = (p.pos(2) - plo[2]) / dx[2];

    int i = static_cast<int>(Math::floor(lx));
    int j = static_cast<int>(Math::floor(ly));
    int k = static_cast<int>(Math::floor(lz));
    Real Fx = fxP * dv;
    Real Fy = fyP * dv;
    Real Fz = fzP * dv;
    Gpu::Atomic::AddNoRet(&ib_fx, Fx);
    Gpu::Atomic::AddNoRet(&ib_fy, Fy);
    Gpu::Atomic::AddNoRet(&ib_fz, Fz);
    RealVect moment = RealVect((p.pos(0) - Px), (p.pos(1) - Py), (p.pos(2) - Pz)).crossProduct(RealVect(Fx, Fy, Fz));
    Gpu::Atomic::AddNoRet(&ib_mx, moment[0]);
    Gpu::Atomic::AddNoRet(&ib_my, moment[1]);
    Gpu::Atomic::AddNoRet(&ib_mz, moment[2]);
    //lagrangian to Euler
    for(int ii = -2; ii < +3; ii++){
        for(int jj = -2; jj < +3; jj++){
            for(int kk = -2; kk < +3; kk ++){
                Real tU, tV, tW;
                const Real xi = (i + ii) * dx[0] + dx[0]/2;
                const Real yj = (j + jj) * dx[1] + dx[1]/2;
                const Real kz = (k + kk) * dx[2] + dx[2]/2;
                deltaFunction( p.pos(0), xi, dx[0], tU, type);
                deltaFunction( p.pos(1), yj, dx[1], tV, type);
                deltaFunction( p.pos(2), kz, dx[2], tW, type);
                Real delta_value = tU * tV * tW;
                Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, k + kk, EulerForceIndex  ), delta_value * Fx);
                Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, k + kk, EulerForceIndex+1), delta_value * Fy);
                Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, k + kk, EulerForceIndex+2), delta_value * Fz);
            }
        }
    }
}

void mParticle::ForceSpreading(MultiFab & EulerForce,
                               kernel& kernel,
                               DELTA_FUNCTION_TYPE type){

    if (verbose) amrex::Print() << "\tmParticle::ForceSpreading\n";
    const auto& gm = mContainer->GetParGDB()->Geom(LOCAL_LEVEL);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.CellSizeArray();

    for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        const auto& particles = pti.GetArrayOfStructs();
        auto Uarray = EulerForce[pti].array();
        auto& attri = pti.GetAttribs();

        auto *const fxP_ptr = attri[P_ATTR::Fx_Marker].data();
        auto *const fyP_ptr = attri[P_ATTR::Fy_Marker].data();
        auto *const fzP_ptr = attri[P_ATTR::Fz_Marker].data();
        const auto *const p_ptr = particles().data();

        auto* loc_ptr = &(kernel.location);
        auto* ib_force_ptr = &(kernel.ib_force);
        auto* moment_ptr = &(kernel.ib_moment);
        auto dv = kernel.dv;
        auto force_index = ParticleProperties::euler_force_index;
        amrex::ParallelFor(np, [=]
        AMREX_GPU_DEVICE (int i) noexcept{
            ForceSpreading_cic(p_ptr[i], (*loc_ptr)[0], (*loc_ptr)[1], (*loc_ptr)[2], fxP_ptr[i], fyP_ptr[i], fzP_ptr[i], 
                              (*ib_force_ptr)[0], (*ib_force_ptr)[1],(*ib_force_ptr)[2], 
                              (*moment_ptr)[0], (*moment_ptr)[1], (*moment_ptr)[2], Uarray, force_index, dv, plo, dxi, type);
        });
    }
    EulerForce.SumBoundary(ParticleProperties::euler_force_index, 3, gm.periodicity());

    if (false) {
        // Check the Multifab
        // Open a file stream for output
        std::ofstream outFile("EulerForce.txt");

        // Check the Multifab
        // for (MFIter mfi(EulerForce, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        for (MFIter mfi(EulerForce, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            outFile << "Box: " << bx << "\n"
                    << "From: (" << bx.smallEnd(0) << ", " << bx.smallEnd(1) << ", " << bx.smallEnd(2) << ") "
                    << "To: (" << bx.bigEnd(0) << ", " << bx.bigEnd(1) << ", " << bx.bigEnd(2) << ")\n";

            Array4<Real> const& a = EulerForce[mfi].array();

            // CPU context or illustrative purposes only
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        // This print statement is for demonstration and should not be used in actual GPU code.
                        outFile << "Processing i: " << i << ", j: " << j << ", k: " << k << " " << a(i,j,k,0) << " " << a(i,j,k,1) << " " << a(i,j,k,2) << "\n";
                    }
                }
            }
        }

        // Close the file when done
        outFile.close();
    }

}

void mParticle::ResetLargrangianPoints(Real dt)
{
    if (verbose) amrex::Print() << "\tmParticle::ResetLargrangianPoints\n";

    for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        auto& attri = pti.GetAttribs();

        auto *const vUP_ptr = attri[P_ATTR::U_Marker].data();
        auto *const vVP_ptr = attri[P_ATTR::V_Marker].data();
        auto *const vWP_ptr = attri[P_ATTR::W_Marker].data();
        auto *const fxP_ptr = attri[P_ATTR::Fx_Marker].data();
        auto *const fyP_ptr = attri[P_ATTR::Fy_Marker].data();
        auto *const fzP_ptr = attri[P_ATTR::Fz_Marker].data();
        amrex::ParallelFor(np, [=]
        AMREX_GPU_DEVICE (int i) noexcept{
            vUP_ptr[i] = 0.0;
            vVP_ptr[i] = 0.0;
            vWP_ptr[i] = 0.0;
            fxP_ptr[i] = 0.0;
            fyP_ptr[i] = 0.0;
            fzP_ptr[i] = 0.0;
        });
    }
}

void mParticle::UpdateParticles(int iStep,
                                const MultiFab& Euler_old, 
                                const MultiFab& Euler,
                                MultiFab& phi_nodal, 
                                MultiFab& pvf, 
                                Real dt)
{
    if (verbose) amrex::Print() << "mParticle::UpdateParticles\n";
    //particle collistion 
    DoParticleCollision(ParticleProperties::collision_model);
    
    MultiFab AllParticlePVF(pvf.boxArray(), pvf.DistributionMap(), pvf.nComp(), pvf.nGrow());
    AllParticlePVF.setVal(0.0);
    
    //continue condition 6DOF
    for(auto& kernel : particle_kernels){

        calculate_phi_nodal(phi_nodal, kernel);
        nodal_phi_to_pvf(pvf, phi_nodal);

        // fixed particle
        if( ( kernel.TL.sum() == 0 ) &&
            ( kernel.RL.sum() == 0 ) ) {
            amrex::Print() << "Particle (" << kernel.id << ") is fixed\n";
            MultiFab::Add(AllParticlePVF, pvf, 0, 0, 1, 0); // do not copy ghost cell values
            continue;
        }

        int ncomp = pvf.nComp();
        int ngrow = pvf.nGrow();
        MultiFab pvf_old(pvf.boxArray(), pvf.DistributionMap(), ncomp, ngrow);
        MultiFab::Copy(pvf_old, pvf, 0, 0, ncomp, ngrow);

        bool at_least_one_free_trans_motion = ( kernel.TL[0] == 2 ) || 
                                              ( kernel.TL[1] == 2 ) ||
                                              ( kernel.TL[2] == 2 );
        bool at_least_one_free_rot_motion   = ( kernel.RL[0] == 2 ) || 
                                              ( kernel.RL[1] == 2 ) ||
                                              ( kernel.RL[2] == 2 );

        int loop = ParticleProperties::loop_solid;

        while (loop > 0 && iStep > ParticleProperties::start_step) {

            if(at_least_one_free_trans_motion) {
                kernel.sum_u_new.scale(0.0);
                kernel.sum_u_old.scale(0.0);
                // sum U
                CalculateSumU_cir(kernel.sum_u_new, Euler, pvf, ParticleProperties::euler_velocity_index);
                CalculateSumU_cir(kernel.sum_u_old, Euler_old, pvf_old, ParticleProperties::euler_velocity_index);
                amrex::ParallelAllReduce::Sum(&kernel.sum_u_new[0], 3, ParallelDescriptor::Communicator());
                amrex::ParallelAllReduce::Sum(&kernel.sum_u_old[0], 3, ParallelDescriptor::Communicator());
            }

            if(at_least_one_free_rot_motion) {
                kernel.sum_t_new.scale(0.0);
                kernel.sum_t_old.scale(0.0);
                // sum T
                CalculateSumT_cir(kernel.sum_t_new, Euler, pvf, kernel.location, ParticleProperties::euler_velocity_index);
                CalculateSumT_cir(kernel.sum_t_old, Euler_old, pvf_old, kernel.location, ParticleProperties::euler_velocity_index);
                amrex::ParallelAllReduce::Sum(&kernel.sum_t_new[0], 3, ParallelDescriptor::Communicator());
                amrex::ParallelAllReduce::Sum(&kernel.sum_t_old[0], 3, ParallelDescriptor::Communicator());
            }

            // 6DOF
            for(auto idir : {0,1,2})
            {
                //TL
                if (kernel.TL[idir] == 0) {
                    kernel.velocity[idir] = 0.0;
                }
                else if (kernel.TL[idir] == 1) {
                    kernel.location[idir] = kernel.location_old[idir] + (kernel.velocity[idir] + kernel.velocity_old[idir]) * dt * 0.5;
                }
                else if (kernel.TL[idir] == 2) {
                    if(!ParticleProperties::Uhlmann){
                        kernel.velocity[idir] = kernel.velocity_old[idir]
                                              + ((kernel.sum_u_new[idir] - kernel.sum_u_old[idir]) * ParticleProperties::euler_fluid_rho / dt 
                                              - kernel.ib_force[idir] * ParticleProperties::euler_fluid_rho
                                              + m_gravity[idir] * (kernel.rho - ParticleProperties::euler_fluid_rho) * kernel.Vp 
                                              + kernel.Fcp[idir]) * dt / kernel.rho / kernel.Vp ;
                    }else{
                        //Uhlmann
                        kernel.velocity[idir] = kernel.velocity_old[idir]
                                              + (ParticleProperties::euler_fluid_rho / kernel.Vp /(ParticleProperties::euler_fluid_rho - kernel.rho)*kernel.ib_force[idir]
                                              + m_gravity[idir]) * dt;
                    }
                    kernel.location[idir] = kernel.location_old[idir] + (kernel.velocity[idir] + kernel.velocity_old[idir]) * dt * 0.5;
                }
                else {
                    amrex::Print() << "Particle (" << kernel.id << ") has wrong TL"<< direction_str[idir] <<" value\n";
                    amrex::Abort("Stop here!");
                }
                //RL
                if (kernel.RL[idir] == 0) {
                    kernel.omega[idir] = 0.0;
                }
                else if (kernel.RL[idir] == 1) {
                }
                else if (kernel.RL[idir] == 2) {
                    if(!ParticleProperties::Uhlmann){
                        kernel.omega[idir] = kernel.omega_old[idir]
                                           + ((kernel.sum_t_new[idir] - kernel.sum_t_old[idir]) * ParticleProperties::euler_fluid_rho / dt
                                           - kernel.ib_moment[idir] * ParticleProperties::euler_fluid_rho
                                           + kernel.Tcp[idir]) * dt / cal_momentum(kernel.rho, kernel.radius);
                    }else{
                        //Uhlmann
                        kernel.omega[idir] = kernel.omega_old[idir]
                                           + ParticleProperties::euler_fluid_rho /(ParticleProperties::euler_fluid_rho - kernel.rho) * kernel.ib_moment[idir] * kernel.dv
                                           / cal_momentum(kernel.rho, kernel.radius) * kernel.rho * dt;
                    }
                }
                else {
                    amrex::Print() << "Particle (" << kernel.id << ") has wrong RL"<< direction_str[idir] <<" value\n";
                    amrex::Abort("Stop here!");
                }

            }
        
            loop--;

            if (loop > 0) {
                calculate_phi_nodal(phi_nodal, kernel);
                nodal_phi_to_pvf(pvf, phi_nodal);
            }

        }
        
        RecordOldValue(kernel);
        MultiFab::Add(AllParticlePVF, pvf, 0, 0, 1, 0); // do not copy ghost cell values
    }
    // calculate the pvf based on the information of all particles
    MultiFab::Copy(pvf, AllParticlePVF, 0, 0, 1, pvf.nGrow());

    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("particle", 4));
}

void mParticle::DoParticleCollision(int model)
{
    if(particle_kernels.size() < 2 ) return ;

    if (verbose) amrex::Print() << "\tmParticle::DoParticleCollision\n";
    
    for(auto kernel : particle_kernels){
        m_Collision.InsertParticle(kernel.location, kernel.velocity, kernel.radius, kernel.rho);
    }
    
    m_Collision.takeModel(model);

    for(auto & particle_kernel : particle_kernels){
        particle_kernel.Fcp = m_Collision.Particles.front().preForece 
                            * particle_kernel.Vp * particle_kernel.rho * m_gravity.vectorLength();
        m_Collision.Particles.pop_front();
    }

}

void mParticle::ComputeLagrangianForce(Real dt, 
                                       const kernel& kernel)
{
    
    if (verbose) amrex::Print() << "\tmParticle::ComputeLagrangianForce\n";

    Real Ub = kernel.velocity[0];
    Real Vb = kernel.velocity[1];
    Real Wb = kernel.velocity[2];
    Real Px = kernel.location[0];
    Real Py = kernel.location[1];
    Real Pz = kernel.location[2];

    for(mParIter pti(*mContainer, LOCAL_LEVEL); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        auto& attri = pti.GetAttribs();
        auto const* p_ptr = pti.GetArrayOfStructs().data();

        auto* Up = attri[P_ATTR::U_Marker].data();
        auto* Vp = attri[P_ATTR::V_Marker].data();
        auto* Wp = attri[P_ATTR::W_Marker].data();
        auto *FxP = attri[P_ATTR::Fx_Marker].data();
        auto *FyP = attri[P_ATTR::Fy_Marker].data();
        auto *FzP = attri[P_ATTR::Fz_Marker].data();

        amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (int i) noexcept{
            auto Ur = (kernel.omega).crossProduct(RealVect(p_ptr[i].pos(0) - Px, p_ptr[i].pos(1) - Py, p_ptr[i].pos(2) - Pz));
            FxP[i] = (Ub + Ur[0] - Up[i])/dt; //
            FyP[i] = (Vb + Ur[1] - Vp[i])/dt; //
            FzP[i] = (Wb + Ur[2] - Wp[i])/dt; //
        });
    }
    if (verbose) mContainer->WriteAsciiFile(amrex::Concatenate("particle", 3));
}

void mParticle::VelocityCorrection(amrex::MultiFab &Euler, amrex::MultiFab &EulerForce, Real dt) const
{
    if(verbose) amrex::Print() << "\tmParticle::VelocityCorrection\n";
    MultiFab::Saxpy(Euler, dt, EulerForce, ParticleProperties::euler_force_index, ParticleProperties::euler_velocity_index, 3, 0); //VelocityCorrection
}

void mParticle::RecordOldValue(kernel& kernel)
{
    kernel.location_old = kernel.location;
    kernel.velocity_old = kernel.velocity;
    kernel.omega_old = kernel.omega;
}

void mParticle::WriteParticleFile(int index)
{
    mContainer->WriteAsciiFile(amrex::Concatenate("particle", index));
}

void mParticle::WriteIBForceAndMoment(int step, amrex::Real time, kernel& current_kernel)
{
    
    if(amrex::ParallelDescriptor::MyProc() != ParallelDescriptor::IOProcessorNumber()) return; 

    std::string file("IB_Particle_" + std::to_string(current_kernel.id) + ".csv");
    std::ofstream out_ib_force;

    std::string head;
    if(!fs::exists(file)){
        head = "iStep,time,X,Y,Z,Vx,Vy,Vz,Rx,Ry,Rz,Fx,Fy,Fz,Mx,My,Mz,Fcpx,Fcpy,Fcpz,Tcpx,Tcpy,Tcpz\n";
    }else{
        head = "";
    }

    out_ib_force.open(file, std::ios::app);
    if(!out_ib_force.is_open()){
        amrex::Print() << "[Particle] write particle file error , step: " << step;
    }else{
        out_ib_force << head << step << "," << time << "," 
                     << current_kernel.location[0] << "," << current_kernel.location[1] << "," << current_kernel.location[2] << ","
                     << current_kernel.velocity[0] << "," << current_kernel.velocity[1] << "," << current_kernel.velocity[2] << ","
                     << current_kernel.omega[0] << "," << current_kernel.omega[1] << "," << current_kernel.omega[2] << ","
                     << current_kernel.ib_force[0] << "," << current_kernel.ib_force[1] << "," << current_kernel.ib_force[2] << "," 
                     << current_kernel.ib_moment[0] << "," << current_kernel.ib_moment[1] << "," << current_kernel.ib_moment[2] << ","
                     << current_kernel.Fcp[0] << "," << current_kernel.Fcp[1] << "," << current_kernel.Fcp[2] << ","
                     << current_kernel.Tcp[0] << "," << current_kernel.Tcp[1] << "," << current_kernel.Tcp[2] << "\n";
    }
    out_ib_force.close();
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                    Particles member function                  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void Particles::create_particles(const Geometry &gm,
                                 const DistributionMapping & dm,
                                 const BoxArray & ba)
{
    amrex::Print() << "[Particle] : create Particle Container\n";
    if(particle->mContainer != nullptr){
        delete particle->mContainer;
        particle->mContainer = nullptr;
    }
    particle->mContainer = new mParticleContainer(gm, dm, ba);

    //get particle tile
    std::pair<int, int> key{0,0};
    auto& particleTileTmp = particle->mContainer->GetParticles(0)[key];
    //insert markers
    if ( ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber() ) {
        //insert particle's markers
        Real phiK = 0;
        for(int marker_index = 0; marker_index < particle->particle_kernels[0].ml; marker_index++){
            //insert code
            mParticleContainer::ParticleType markerP;
            markerP.id() = marker_index + 1;
            markerP.cpu() = ParallelDescriptor::MyProc();
            markerP.pos(0) = particle->particle_kernels[0].location[0];
            markerP.pos(1) = particle->particle_kernels[0].location[1];
            markerP.pos(2) = particle->particle_kernels[0].location[2];

            std::array<ParticleReal, numAttri> Marker_attr;
            Marker_attr[U_Marker] = 0.0;
            Marker_attr[V_Marker] = 0.0;
            Marker_attr[W_Marker] = 0.0;
            Marker_attr[Fx_Marker] = 0.0;
            Marker_attr[Fy_Marker] = 0.0;
            Marker_attr[Fz_Marker] = 0.0;

            particleTileTmp.push_back(markerP);
            particleTileTmp.push_back_real(Marker_attr);
        }
    }
    particle->mContainer->Redistribute(); // Still needs to redistribute here! 

    ParticleProperties::plo = gm.ProbLoArray();
    ParticleProperties::phi = gm.ProbHiArray();
    ParticleProperties::dx = gm.CellSizeArray();
}

mParticle* Particles::get_particles()
{
    return particle;
}


void Particles::init_particle(Real gravity, Real h)
{
    amrex::Print() << "[Particle] : create Particle's kernel\n";
    particle = new mParticle;
    if(particle != nullptr){
        isInitial = true;
        particle->InitParticles(
            ParticleProperties::_x, 
            ParticleProperties::_y, 
            ParticleProperties::_z,
            ParticleProperties::_rho,
            ParticleProperties::Vx,
            ParticleProperties::Vy,
            ParticleProperties::Vz,
            ParticleProperties::Ox,
            ParticleProperties::Oy,
            ParticleProperties::Oz,
            ParticleProperties::TLX,
            ParticleProperties::TLY,
            ParticleProperties::TLZ,
            ParticleProperties::RLX,
            ParticleProperties::RLY,
            ParticleProperties::RLZ,
            ParticleProperties::_radius,
            h,
            gravity,
            ParticleProperties::verbose);
    }

}

void Particles::Initialize()
{
    ParmParse pp("particle");

    std::string particle_inputfile;

    pp.get("input",particle_inputfile);
    
    if(!particle_inputfile.empty()){
        ParmParse p_file(particle_inputfile);
        p_file.getarr("x",          ParticleProperties::_x);
        p_file.getarr("y",          ParticleProperties::_y);
        p_file.getarr("z",          ParticleProperties::_z);
        p_file.getarr("rho",        ParticleProperties::_rho);
        p_file.getarr("velocity_x", ParticleProperties::Vx);
        p_file.getarr("velocity_y", ParticleProperties::Vy);
        p_file.getarr("velocity_z", ParticleProperties::Vz);
        p_file.getarr("omega_x",    ParticleProperties::Ox);
        p_file.getarr("omega_y",    ParticleProperties::Oy);
        p_file.getarr("omega_z",    ParticleProperties::Oz);
        p_file.getarr("TLX",        ParticleProperties::TLX);
        p_file.getarr("TLY",        ParticleProperties::TLY);
        p_file.getarr("TLZ",        ParticleProperties::TLZ);
        p_file.getarr("RLX",        ParticleProperties::RLX);
        p_file.getarr("RLY",        ParticleProperties::RLY);
        p_file.getarr("RLZ",        ParticleProperties::RLZ);
        p_file.getarr("radius",     ParticleProperties::_radius);
        p_file.query("LOOP_NS",     ParticleProperties::loop_ns);
        p_file.query("LOOP_SOLID",  ParticleProperties::loop_solid);
        p_file.query("verbose",     ParticleProperties::verbose);
        p_file.query("start_step",  ParticleProperties::start_step);
        p_file.query("Uhlmann",     ParticleProperties::Uhlmann);
        p_file.query("collision_model", ParticleProperties::collision_model);
        
        ParmParse ns("ns");
        ns.get("fluid_rho",      ParticleProperties::euler_fluid_rho);
        
        ParmParse level_parse("amr");
        level_parse.get("max_level", ParticleProperties::euler_finest_level);

        ParmParse geometry_parse("geometry");
        geometry_parse.getarr("prob_lo", ParticleProperties::GLO);
        geometry_parse.getarr("prob_hi", ParticleProperties::GHI);
        amrex::Print() << "[Particle] : Reading partilces cfg file : " << particle_inputfile << "\n"
                       << "             Particle's level : " << ParticleProperties::euler_finest_level << "\n";
    }else {
        amrex::Abort("[Particle] : can't read particles settings, pls check your config file \"particle.input\"");
    }
}

int Particles::ParticleFinestLevel()
{
    return ParticleProperties::euler_finest_level;
}
