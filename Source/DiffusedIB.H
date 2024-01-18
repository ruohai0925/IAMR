#ifndef DIFFUSED_IB_H
#define DIFFUSED_IB_H

#include <AMReX_Particle.H>
#include <AMReX_ParticleContainer.H>

namespace amrex{
using Location = XDim3;
using ParticlesLoc = amrex::Vector<Location>;
using Markers = ParticlesLoc;
using ParticlesRadious = amrex::Vector<amrex::Real>;

using MarkerInfo = struct {XDim3 loc; amrex::XDim3 velocity;};
using MarkersInfo = amrex::Vector<MarkerInfo>;

enum ParticleFiled{
    Vx = 0,
    Vy,
    Vz,
    Fx,
    Fy,
    Fz,
    NUM_OF_COMPONENTS
};

const int realData = NUM_OF_COMPONENTS;
const int intData  = 2;

const int marker = 20;

template<int REAL_DATA = realData, int INT_DATA = intData>
class MyParticle// : public amrex::Particle<3, 2>
{
public:
    using Particle = amrex::Particle<REAL_DATA,INT_DATA>;
    using ParticleMarkers = amrex::Vector<Particle>;

    explicit MyParticle(Location location, Real radious, int markNum){
        mParticle.id() = Particle::NextID();
        //cpu's infomation
        //mParticle.cpu() = xxxx;
        mParticle.pos(0) = location.x;
        mParticle.pos(1) = location.y;
        mParticle.pos(2) = location.z;

        Real deltaAngle = 360.0 / markNum;
        for(::size_t index = 0; index < markNum; index++){
            //2D , coordinates z = z;
            Particle p;
            p.cpu() = mParticle.m_idcpu;
            p.pos(0) = mParticle.m_pos(0) + ::std::sin((double)index * deltaAngle);
            p.pos(1) = mParticle.m_pos(1) + ::std::cos((double)index * deltaAngle);
            p.pos(2) = mParticle.m_pos(2);
            p.rdata(Vx) = 0.0;
            p.rdata(Vy) = 0.0;
            p.rdata(Vz) = 0.0;
            p.rdata(Fx) = 0.0;
            p.rdata(Fy) = 0.0;
            p.rdata(Fz) = 0.0;
            mMarkers.emplace_back(p);
        }
    }

    Particle& getParticle(){
        return mParticle;
    }

    ParticleMarkers& getMarkers(){
        return mMarkers;
    }

private:
    Real                  mRadious;
    Particle              mParticle;
    ParticleMarkers       mMarkers;
    amrex::ParticleContainer<REAL_DATA, INT_DATA, 4, 4> container;
};

/**
 * mesh's information and the particle's infomation with the markers
 *
 * 1.Radius 2.largrangian point's location 3.velocity
 *
 * CPU's infomation (Thread rank, size)
 */
class Diffused_IB{
public:
    Diffused_IB() = delete;

    Diffused_IB(MultiFab& phi, Geometry& gm);

    explicit Diffused_IB(const Diffused_IB& other);
    explicit Diffused_IB(Diffused_IB&& other) noexcept;
    ~Diffused_IB();

    Diffused_IB& operator=(const Diffused_IB& other);
    Diffused_IB& operator=(Diffused_IB&& other) noexcept;

    // void InitParticlesAndMarkers(const ParticlesLoc& Locations, const ParticlesRadious& Radious);
    void VelocityInterpolation();
    void ComputeLagrangianForce();
    void ForceSpreading();
    void VelocityCorrection();


private:
    Vector<MyParticle<>> mSolid;
    // amrex::ParticleContainer<3, 2, 4, 4> container;

};

}
#endif
