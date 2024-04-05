
#include <DiffusedIB.H>

#include <AMReX_ParmParse.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_FillPatchUtil.H>
#include <iamr_constants.H>

using namespace amrex;

void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal)
{

    amrex::Print() << "In the nodal_phi_to_pvf " << std::endl;

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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                     other function                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

[[nodiscard]] AMREX_FORCE_INLINE
Real cal_momentum(Real rho, Real radious)
{
    return 8.0 * Math::pi<Real>() * rho * Math::powi<5>(radious) / 15.0;
}

AMREX_FORCE_INLINE
void deltaFunction(Real xf, Real xp, Real h, Real& value, DELTA_FUNCTION_TYPE type)
{
    Real rr = amrex::Math::abs(( xf - xp ) / h);

    switch (type) {
    case DELTA_FUNCTION_TYPE::FOUR_POINT_IB:
        if(rr >= 0 && rr < 0.5 ){
            value = 1.0 / 8.0 * ( 3.0 - 2.0 * rr + std::sqrt( 1.0 + 4.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else if (rr >= 1 && rr < 2) {
            value = 1.0 / 8.0 * ( 5.0 - 2.0 * rr - std::sqrt( -7.0 + 12.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    case DELTA_FUNCTION_TYPE::THREE_POINT_IB:
        if(rr >= 0 && rr < 1){
            value = 1.0 / 6.0 * ( 5.0 - 3.0 * rr + std::sqrt( - 3.0 * ( 1 - Math::powi<2>(rr)) + 1.0 )) / h;
        }else if (rr >= 1 && rr < 2) {
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
void mParticle::InteractWithEuler(MultiFab &EulerVel, MultiFab &EulerForce, int loop_time, Real dt, Real alpha_k, DELTA_FUNCTION_TYPE type){

    if (verbose) amrex::Print() << "mParticle::InteractWithEuler " << std::endl;

    for(kernel& kernel : particle_kernels){
        
        InitialWithLargrangianPoints(kernel); // Initialize markers for a specific particle
        
        //UpdateParticles(Euler, kernel, dt, alpha_k);
        const int EulerForceIndex = euler_force_index;
        //for 1 -> Ns
        while(loop_time > 0){
            EulerForce.setVal(0.0, EulerForceIndex, 3, EulerForce.nGrow()); //clear Euler force
            VelocityInterpolation(EulerVel, type);
            ComputeLagrangianForce(dt, kernel);
            ForceSpreading(EulerForce, type);
            MultiFab::Saxpy(EulerVel, dt, EulerForce, euler_force_index, euler_velocity_index, 3, 0); //VelocityCorrection
            loop_time--;
        };
    }
}

void mParticle::InitParticles(const Vector<Real>& x,
                              const Vector<Real>& y,
                              const Vector<Real>& z,
                              Real rho_s,
                              Real radious,
                              Real rho_f, 
                              int force_index, 
                              int velocity_index,
                              int finest_level){
    
    if (verbose) amrex::Print() << "mParticle::InitParticles " << std::endl;
    
    euler_finest_level = finest_level;                                      
    euler_force_index = force_index;
    euler_fluid_rho = rho_f;
    euler_velocity_index = velocity_index;

    // Assuming the variables are defined similarly
    // amrex::Print() << "euler_finest_level: " << euler_finest_level << "\n"
    //             << "euler_force_index: " << euler_force_index << "\n"
    //             << "euler_fluid_rho: " << euler_fluid_rho << "\n"
    //             << "euler_velocity_index: " << euler_velocity_index << "\n";

    //pre judge
    if(!((x.size() == y.size()) && (x.size() == z.size()))){
        Print() << "particle's position container are all different size";
        return;
    }
    //all the particles have same radious
    Real phiK = 0;
    Real h = m_gdb->Geom(euler_finest_level).CellSizeArray()[0];
    int Ml = static_cast<int>( Math::pi<Real>() / 3 * (12 * Math::powi<2>(radious / h)));
    Real dv = Math::pi<Real>() * h / 3 / Ml * (12 * radious * radious + h * h);

    if (verbose) amrex::Print() << "h: " << h << ", Ml: " << Ml << ", dv: " << dv << "\n";

    for(int index = 0; index < x.size(); index++){
        kernel mKernel;
        mKernel.location[0] = x[index];
        mKernel.location[1] = y[index];
        mKernel.location[2] = z[index];
        mKernel.velocity[0] = 0.0;
        mKernel.velocity[1] = 0.0;
        mKernel.velocity[2] = 0.0;
        mKernel.omega[0] = 0.0;
        mKernel.omega[1] = 0.0;
        mKernel.omega[2] = 0.0;
        mKernel.varphi[0] = 0.0;
        mKernel.varphi[1] = 0.0;
        mKernel.varphi[2] = 0.0;
        mKernel.radious = radious;
        mKernel.ml = Ml;
        mKernel.dv = dv;
        mKernel.rho = rho_s;
        particle_kernels.push_back(mKernel);

        if (verbose) amrex::Print() << "Kernel " << index << ": Location (" << x[index] << ", " << y[index] << ", " << z[index] 
                   << "), Radius: " << radious << ", Ml: " << Ml << ", dv: " << dv << ", Rho: " << rho_s << "\n";
    }

    //get particle tile
    std::pair<int, int> key{0,0};
    auto& particleTileTmp = GetParticles(0)[key];

    //insert markers
    if ( ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber() ) {
        //insert particle's markers
        for(int marker_index = 0; marker_index < Ml; marker_index++){
            //insert code
            ParticleType markerP;
            markerP.id() = ParticleType::NextID();
            markerP.cpu() = ParallelDescriptor::MyProc();
            markerP.pos(0) = 0;
            markerP.pos(1) = 0;
            markerP.pos(2) = 0;

            std::array<ParticleReal, numAttri> Marker_attr;
            Marker_attr[U_Marker] = 0.0;
            Marker_attr[V_Marker] = 0.0;
            Marker_attr[W_Marker] = 0.0;
            Marker_attr[Fx_Marker] = 0.0;
            Marker_attr[Fy_Marker] = 0.0;
            Marker_attr[Fz_Marker] = 0.0;
            // attr[V] = 10.0;
            particleTileTmp.push_back(markerP);
            particleTileTmp.push_back_real(Marker_attr);
        }
    }
    Redistribute(); // No need to redistribute here! 
    WriteAsciiFile(amrex::Concatenate("particle", 0));
}

void mParticle::InitialWithLargrangianPoints(const kernel& current_kernel){

    if (verbose) amrex::Print() << "mParticle::InitialWithLargrangianPoints " << std::endl;

    // Update the markers' locations
    {
        mParIter pti(*this, euler_finest_level);
        auto *particles = pti.GetArrayOfStructs().data();

        Real phiK = 0;
        for(int index = 0; index < current_kernel.ml; index++){
            Real Hk = -1.0 + 2.0 * (index) / ( current_kernel.ml - 1.0);
            Real thetaK = std::acos(Hk);
            if(index == 0 || index == ( current_kernel.ml - 1)){
                phiK = 0;
            }else {
                phiK = std::fmod( phiK + 3.809 / std::sqrt(current_kernel.ml) / std::sqrt( 1 - Math::powi<2>(Hk)) , 2 * Math::pi<Real>());
            }
            // update LargrangianPoint position with particle position           
            particles[index].pos(0) = current_kernel.location[0] + current_kernel.radious * std::sin(thetaK) * std::cos(phiK);
            particles[index].pos(1) = current_kernel.location[1] + current_kernel.radious * std::sin(thetaK) * std::sin(phiK);
            particles[index].pos(2) = current_kernel.location[2] + current_kernel.radious * std::cos(thetaK);
        }
    }
    // Redistribute the markers after updating their locations
    Redistribute();
    WriteAsciiFile(amrex::Concatenate("particle", 1));
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
                Gpu::Atomic::AddNoRet( &Up, delta_value * E(i + ii, j + jj, k + kk, EulerVIndex  ) * d );
                Gpu::Atomic::AddNoRet( &Vp, delta_value * E(i + ii, j + jj, k + kk, EulerVIndex+1) * d );
                Gpu::Atomic::AddNoRet( &Wp, delta_value * E(i + ii, j + jj, k + kk, EulerVIndex+2) * d );
            }
        }
    }
}

void mParticle::VelocityInterpolation(const MultiFab &EulerVel,
                                      DELTA_FUNCTION_TYPE type)//
{

    if (verbose) amrex::Print() << "mParticle::VelocityInterpolation " << std::endl;

    //amrex::Print() << "euler_finest_level " << euler_finest_level << std::endl;

    const auto& gm = m_gdb->Geom(euler_finest_level);
    auto plo = gm.ProbLoArray();
    auto phi = gm.ProbHiArray();
    auto dx = gm.CellSizeArray();
    const int EulerVelocityIndex = euler_velocity_index;

    // std::cout << "plo: ";
    // for (const auto& val : plo) std::cout << val << " ";
    // std::cout << "phi: ";
    // for (const auto& val : phi) std::cout << val << " ";
    // std::cout << "\ndx: ";
    // for (const auto& val : dx) std::cout << val << " ";
    // std::cout << "\nEulerVelocityIndex: " << EulerVelocityIndex << std::endl;

    auto ba = EulerVel.boxArray();
    // amrex::Print() << "ba " << ba << std::endl;

    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        
        const Box& box = pti.validbox();
        // std::cout << "box: " << box << std::endl;
        
        auto& particles = pti.GetArrayOfStructs();
        auto *p_ptr = particles.data();
        const Long np = pti.numParticles();

        // std::cout << "Particles count (np): " << np << std::endl;

        auto& attri = pti.GetAttribs();
        auto* Up = attri[P_ATTR::U_Marker].data();
        auto* Vp = attri[P_ATTR::V_Marker].data();
        auto* Wp = attri[P_ATTR::W_Marker].data();
        // const auto& E = EulerVel[pti].array();
        const auto& E = EulerVel.array(pti);

        //std::cout << "Attributes (attri): " << &attri << std::endl; // Placeholder for actual method to print or summarize 'attri'

        //std::cout << "Up, Vp, Wp pointers: " << Up << ", " << Vp << ", " << Wp << std::endl;

        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            VelocityInterpolation_cir(i, p_ptr[i], Up[i], Vp[i], Wp[i], E, EulerVelocityIndex, box.loVect(), box.hiVect(), plo, dx, type);
        });
    }
    WriteAsciiFile(amrex::Concatenate("particle", 2));
    //amrex::Abort("stop here!");
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ForceSpreading_cic (P const& p,
                  ParticleReal fxP,
                  ParticleReal fyP,
                  ParticleReal fzP,
                  Array4<Real> const& E,
                  int EulerForceIndex,
                  GpuArray<Real,AMREX_SPACEDIM> const& plo,
                  GpuArray<Real,AMREX_SPACEDIM> const& dx,
                  DELTA_FUNCTION_TYPE type)
{
    const Real d = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    //plo to ii jj kk
    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    Real lz = (p.pos(2) - plo[2]) / dx[2];

    int i = static_cast<int>(Math::floor(lx));
    int j = static_cast<int>(Math::floor(ly));
    int k = static_cast<int>(Math::floor(lz));
    // calc_delta(i, j, k, dxi, rho);
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
                Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, k + kk, EulerForceIndex  ), delta_value * fxP * d);
                Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, k + kk, EulerForceIndex+1), delta_value * fyP * d);
                Gpu::Atomic::AddNoRet(&E(i + ii, j + jj, k + kk, EulerForceIndex+2), delta_value * fzP * d);
            }
        }
    }
}

void mParticle::ForceSpreading(MultiFab & EulerForce, 
                               DELTA_FUNCTION_TYPE type){

    if (verbose) amrex::Print() << "mParticle::ForceSpreading " << std::endl;

    int index = 0;
    const auto& gm = m_gdb->Geom(euler_finest_level);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.CellSizeArray();
    const int EulerForceIndex = euler_force_index;
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        const Long np = pti.numParticles();
        const auto& fxP = pti.GetStructOfArrays().GetRealData(P_ATTR::U_Marker);//Fx_Marker 
        const auto& fyP = pti.GetStructOfArrays().GetRealData(P_ATTR::V_Marker);//Fy_Marker 
        const auto& fzP = pti.GetStructOfArrays().GetRealData(P_ATTR::W_Marker);//Fz_Marker 
        const auto& particles = pti.GetArrayOfStructs();

        auto Uarray = EulerForce[pti].array();

        const auto& fxP_ptr = fxP.data();
        const auto& fyP_ptr = fyP.data();
        const auto& fzP_ptr = fzP.data();
        const auto& p_ptr = particles().data();
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            ForceSpreading_cic(p_ptr[i], fxP_ptr[i], fyP_ptr[i], fzP_ptr[i], Uarray, EulerForceIndex, plo, dxi, type);
        });
    }
    EulerForce.SumBoundary(EulerForceIndex, 3, gm.periodicity());
    
    // Check the Multifab
    // Open a file stream for output
    std::ofstream outFile("EulerForce.txt");

    // Check the Multifab
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
                    outFile << "Processing i: " << i << ", j: " << j << ", k: " << k << " " << a(i,j,k,0) << " " << a(i,j,k,1) << " " << a(i,j,k,2) << std::endl;
                }
            }
        }
    }

    // Close the file when done
    outFile.close();

}

void mParticle::UpdateParticles(const amrex::MultiFab& Euler, kernel& kernel, Real dt, Real alpha_k)
{
    
    if (verbose) amrex::Print() << "mParticle::UpdateParticles " << std::endl;
    
    const auto& gm = m_gdb->Geom(euler_finest_level);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.InvCellSizeArray();
    //update the kernel's infomation and cal body force
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        auto &particles = pti.GetArrayOfStructs();
        auto *p_ptr = particles.data();
        auto &attri = pti.GetAttribs();
        auto *FxP = attri[P_ATTR::Fx_Marker].data();
        auto *FyP = attri[P_ATTR::Fy_Marker].data();
        auto *FzP = attri[P_ATTR::Fz_Marker].data();
        auto *UP  = attri[P_ATTR::U_Marker].data();
        auto *VP  = attri[P_ATTR::V_Marker].data();
        auto *WP  = attri[P_ATTR::W_Marker].data();
        const Real Dv = kernel.dv;
        const Long np = pti.numParticles();
        RealVect ForceDv{std::vector<Real>{0.0,0.0,0.0}};
        RealVect Moment{std::vector<Real>{0.0,0.0,0.0}};
        auto *ForceDv_ptr  = &ForceDv;
        auto *Moment_ptr   = &Moment;
        auto *location_ptr = &kernel.location;
        auto *omega_ptr    = &kernel.omega;
        auto *velocity_ptr = &kernel.velocity;
        auto *varphi_ptr   = &kernel.varphi;
        const Real rho_p = kernel.rho;
        //sum
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            //calculate the force
            //find current particle's lagrangian marker
            *ForceDv_ptr += RealVect(AMREX_D_DECL(FxP[i],FyP[i],FzP[i])) * Dv;
            *Moment_ptr +=  (RealVect(AMREX_D_DECL(p_ptr[i].pos(0),p_ptr[i].pos(1),p_ptr[i].pos(2))) - *location_ptr).crossProduct(
                            RealVect(AMREX_D_DECL(FxP[i],FyP[i],FzP[i]))) * Dv;
        });
        RealVect oldVelocity = kernel.velocity;
        RealVect oldOmega = kernel.omega;
        kernel.velocity = kernel.velocity -
                            2 * alpha_k * dt/( Math::pi<Real>() * 4 * Math::powi<3>(kernel.radious) / 3) / (kernel.rho - euler_fluid_rho) * (ForceDv);// + mVector(0.0, -9.8, 0.0));
        kernel.omega = kernel.omega -
                        2 * alpha_k * dt * kernel.rho / cal_momentum(kernel.rho, kernel.radious) / (kernel.rho - euler_fluid_rho) * Moment;
        
        auto deltaX = alpha_k * dt * (*velocity_ptr + oldVelocity);
        *location_ptr = *location_ptr + deltaX;
        *varphi_ptr = *varphi_ptr + alpha_k * dt * (*omega_ptr + oldOmega);
        //sum
        auto Uarray = Euler[pti].array();
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            //calculate the force
            //find current particle's lagrangian marker
            RealVect tmp = (*omega_ptr).crossProduct(*location_ptr - RealVect(p_ptr[i].pos(0),
                                                                                    p_ptr[i].pos(1),
                                                                                    p_ptr[i].pos(2)));
            FxP[i] = rho_p / dt *(UP[i] + tmp[0]);
            FyP[i] = rho_p / dt *(VP[i] + tmp[1]);
            FzP[i] = rho_p / dt *(WP[i] + tmp[2]);
            p_ptr[i].pos(0) += deltaX[0];
            p_ptr[i].pos(1) += deltaX[1];
            p_ptr[i].pos(2) += deltaX[2];
        });
    }
    WriteAsciiFile(amrex::Concatenate("particle", 4));
}

void mParticle::ComputeLagrangianForce(Real dt, const kernel& kernel)
{
    
    if (verbose) amrex::Print() << "mParticle::ComputeLagrangianForce " << std::endl;

    Real Ub = kernel.velocity[0];
    Real Vb = kernel.velocity[1];
    Real Wb = kernel.velocity[2];

    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        auto& particles = pti.GetArrayOfStructs();
        auto *p_ptr = particles.data();
        const Long np = pti.numParticles();

        auto& attri = pti.GetAttribs();
        auto* Up = attri[P_ATTR::U_Marker].data();
        auto* Vp = attri[P_ATTR::V_Marker].data();
        auto* Wp = attri[P_ATTR::W_Marker].data();
        auto *FxP = attri[P_ATTR::Fx_Marker].data();
        auto *FyP = attri[P_ATTR::Fy_Marker].data();
        auto *FzP = attri[P_ATTR::Fz_Marker].data();
        amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (int i) noexcept{
            FxP[i] = (Ub - Up[i])/dt; //
            FyP[i] = (Vb - Vp[i])/dt; //
            FzP[i] = (Wb - Wp[i])/dt; //
        });
    }
    WriteAsciiFile(amrex::Concatenate("particle", 3));
}

void mParticle::WriteParticleFile(int index)
{
    WriteAsciiFile(amrex::Concatenate("particle", index));
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                    Particles member function                  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void Particles::create_particles(const Geometry &gm,
                             const DistributionMapping & dm,
                             const BoxArray & ba)
{
    particle = new mParticle(gm, dm, ba);
}

mParticle* Particles::get_particles()
{
    return particle;
}

void Particles::define_para(const Vector<Real>& x,
                            const Vector<Real>& y,
                            const Vector<Real>& z,
                            Real rho_s,
                            Real radious,
                            Real rho_f, 
                            int force_index, 
                            int velocity_index,
                            int finest_level)
{
    std::copy(x.begin(), x.end(), std::back_inserter(_x));
    std::copy(y.begin(), y.end(), std::back_inserter(_y));
    std::copy(z.begin(), z.end(), std::back_inserter(_z));

    euler_fluid_rho = rho_f;
    euler_solid_rho = rho_s;

    particle_radious = radious;

    euler_finest_level = finest_level;
    euler_velocity_index = velocity_index;
    euler_force_index = force_index;
}

void Particles::init_particle()
{
    if(particle != nullptr){
        particle->InitParticles(_x, _y, _z, euler_solid_rho, particle_radious, euler_fluid_rho,
         euler_force_index, euler_velocity_index, euler_finest_level);
    }
}