#include "DiffusedIB.H"

namespace amrex{

Diffused_IB::Diffused_IB(MultiFab& phi, Geometry& gm){
    //parse the infomation from file 
    ParmParse pp("particle");
    int particleNum{0};
    pp.query("tot", particleNum);
    Real radious{0.0};
    pp.query("radious", radious);
    std::string locationFile;
    pp.query("file", locationFile);

    std::ifstream location(locationFile);
    std::string tmpline;
    while (location.eof()) {
        std::getline(location, tmpline);
        std::string x, y;
        std::stringstream tmp(tmpline);
        tmp >> x >> y;
        mSolid.emplace_back(MyParticle<>({std::atof(x.c_str()), std::atof(y.c_str()), 0.0}, radious, marker));
    }

    // get information from multiFab
    const auto& ba = phi.boxArray();
    const auto& box = gm.ProbDomain();
    const auto& lo = gm.ProbLoArray();
    const auto& hi = gm.ProbHiArray();
    const DistributionMapping & dm = phi.DistributionMap();

#ifdef AMREX_USE_OMP
#pargma omp parallel if (Gpu::notInLaunchRegion())
#endif
    
    for( MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi){
        amrex::ParallelFor(mfi.tilebox(), [box, lo, hi, this] AMREX_GPU_DEVICE(int i, int j, int k) noexcept{
            auto const& hight = box.hi();
            for(auto& item : mSolid){
                for(auto& marker : item.getMarkers()){
                    //compare mesh coordinate with marker position
                    // if(marker.pos(0))
                }
            }
        });
    }
}

}