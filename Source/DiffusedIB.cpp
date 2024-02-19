#include "DiffusedIB.H"

using namespace amrex;

void calc_delta(int i, int j, int k, int di, int dj, int dk, amrex::Array4<amrex::Real> const& delta)
{

}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void deposit_cic (P const& p, amrex::ParticleReal wp, amrex::Real charge,
                  amrex::Array4<amrex::Real> const& rho,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
{
    amrex::Real inv_vol = AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]);

    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0];
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1];
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2];

    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));

    amrex::Real wx_hi = lx - i;
    amrex::Real wy_hi = ly - j;
    amrex::Real wz_hi = lz - k;

    amrex::Real wx_lo = amrex::Real(1.0) - wx_hi;
    amrex::Real wy_lo = amrex::Real(1.0) - wy_hi;

    amrex::Real qp = wp + i + j + k;

    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j,   k  , 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j+1, k  , 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j,   k  , 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j+1, k  , 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j,   k+1, 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i,   j+1, k+1, 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j,   k+1, 0), qp);
    amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j+1, k+1, 0), qp);
    // amrex::Gpu::Atomic::AddNoRet(&rho(i,   j,   i, 1), qp);
    // amrex::Gpu::Atomic::AddNoRet(&rho(i,   j+1, i, 1), qp);
    // amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j,   i, 1), qp);
    // amrex::Gpu::Atomic::AddNoRet(&rho(i+1, j+1, i, 1), qp);
}

/**
 * @brief 
 * interpolate the Eular filed to particle attributes
 * @tparam P particle container type
 * @param p partilce container pointer
 * @param Tp particle attributes T
 * @param Vp particle attributes V
 * @param T Eular attributes T
 * @param V Eular attributes V
 * @param plo low tri
 * @param dxi dx dy dz [0,1,2]
 * @return AMREX_GPU_HOST_DEVICE 
 */
template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void interpolate_cir(P const& p, amrex::Real& Tp, amrex::Real& Vp,
                     amrex::Array4<amrex::Real const> const& E,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi)
{
    const amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0]; // x
    const amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1]; // y
    const amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2]; // z

    int i = static_cast<int>(amrex::Math::floor(lx));// i
    int j = static_cast<int>(amrex::Math::floor(ly));// j
    int k = static_cast<int>(amrex::Math::floor(lz));// k

    Tp = (
          E(i - 1, j, k, 0) +
          E(i, j - 1, k, 0) +
          E(i, j, k - 1, 0)
    ) / 6;

    // Vp = (
    //       E(i + 1, j, k, 1) + E(i - 1, j, k, 1) +
    //       E(i, j + 1, k, 1) + E(i, j - 1, k, 1) +
    //       E(i, j, k + 1, 1) + E(i, j, k - 1, 1)
    // ) / 6;
}

/* * * * * * * * * * * * * * * * * * particle code * * * * * * * * * * * * * * * * * * * * * */

void mParticle::initParticle(){

    if ( ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber() ) {
        ParticleType p;
        p.id() = ParticleType::NextID();
        p.cpu() = ParallelDescriptor::MyProc();

        p.pos(0) = 0.5;
        p.pos(1) = 0.5;
        p.pos(2) = 0.5;

        std::array<ParticleReal, 2> attr;
        attr[T] = 2.0;
        // attr[V] = 10.0;

        // ParticleType p2;
        // p2.id() = ParticleType::NextID();
        // p2.cpu() = ParallelDescriptor::MyProc();

        // p2.pos(0) = 0.5;
        // p2.pos(1) = 0.5;
        // p2.pos(2) = 0.5;

        // std::array<ParticleReal, 2> attr2;
        // attr2[T] = 12.5;
        // attr2[V] = 10.0;

        //

        std::pair<int, int> key{0,0};
        auto& particleTileTmp = GetParticles(0)[key];
        particleTileTmp.push_back(p);
        // particleTileTmp.push_back(p2);
        particleTileTmp.push_back_real(attr);
        // particleTileTmp.push_back_real(attr2);

        WriteAsciiFile(amrex::Concatenate("particle", 0));
    }
        Redistribute();
}

void mParticle::DepositChange(MultiFab & tU){
    int fine_level = 0;
    amrex::Print() << "fine_level : " << fine_level
                    << "tU's size : " << tU.size()
                    << "tU n grow : " << tU.nGrow();
    const int ng = tU.nGrow();
    int index = 0;
    const auto& gm = m_gdb->Geom(fine_level);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.InvCellSizeArray();
    for(mParIter pti(*this, 0); pti.isValid(); ++pti){
        //gemotry infomation
        amrex::Print() << "current index : " << index++;


        const Long np = pti.numParticles();
        const auto& wp = pti.GetStructOfArrays().GetRealData(T);
        const auto& particles = pti.GetArrayOfStructs();
        auto Uarray = tU[pti].array();

        const auto& wp_ptr = wp.data();
        const auto& p_ptr = particles().data();
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept{
            deposit_cic(p_ptr[i], wp_ptr[i], 1.0, Uarray, plo, dxi);
        });
    }

        //gemotry infomation
    amrex::Print() << "current index : " << index++;
}


void mParticle::FiledGather(const amrex::MultiFab &Eular)
{
    const int ng = Eular.nGrow();
    const int level = 0;
    const auto& gm = m_gdb->Geom(level);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.InvCellSizeArray();
    //assert
    AMREX_ASSERT(OnSameGrids(level, *E[0]));

    for(mParIter pti(*this, 0); pti.isValid(); ++pti){
        auto& particles = pti.GetArrayOfStructs();
        auto p_ptr = particles.data();
        const Long np = pti.numParticles();

        auto& attri = pti.GetAttribs();
        auto Tp = attri[T].data();
        auto Vp = attri[V].data();

        const auto& E = Eular[pti].array();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept{
            interpolate_cir(p_ptr[i], Tp[i], Vp[i], E, plo, dxi);
        });
    }

    WriteAsciiFile(amrex::Concatenate("particle", 1));
}
