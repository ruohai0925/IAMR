#include <AMReX_Particles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Array.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace amrex {

enum typeC{
    T = 0,
    V,
    num
};

class mParIter : public ParIter<0, 0, 2>{
public:
    using amrex::ParIter<0, 0, 2>::ParIter;
    using RealVector = amrex::ParIter<0, 0, 2>::ContainerType::RealVector;
 
    const std::array<RealVector, 2>& GetAttribs () const {
        return GetStructOfArrays().GetRealData();
    }

    std::array<RealVector, 2>& GetAttribs () {
        return GetStructOfArrays().GetRealData();
    }

    const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

class mParticle : public ParticleContainer<0, 0, 2>{
public:
    explicit mParticle(const Vector<Geometry>            & gm,
                       const Vector<DistributionMapping> & dm,
                       const Vector<BoxArray>            & ba,
                       const Vector<int>                 & rr): ParticleContainer<0, 0, 2>(gm, dm, ba, rr){}
    explicit mParticle(const Geometry &gm,
                       const DistributionMapping & dm,
                       const BoxArray & ba)
                       : ParticleContainer<0, 0, 2>(gm, dm, ba){}
    void initParticle();

    void DepositChange(MultiFab & tU);

    void FiledGather(const amrex::MultiFab &Eular);
};
}