#include <omp.h>
#include <string>
#include <iostream>
#include "ms.h"
#include "grid.h"
#include "mrcio.h"
#include "pdbio.h"
#include "parser.h"
using namespace std;

const char* help = "Find local maxima of density map and convert to points\n"
"Usage: getp --in input.mrc --out output.pdb \n"
"           [--thresh [thresold of map, default=5.0]] \n"
"           [--res [resolution, default=6.0]] \n"
"           [--nt  [num of threads, default=4]] \n"
"           [--rmax [max radiu to shift, default=10]] \n"
"           [--dmerge [merge distance, default=1.0]] \n"
"           [--pdb [input initial coordinates in .pdb format]] \n"
;

int main(int argc, char* argv[]){
    // Read args
    auto args = parse_args(argc, argv);
    auto pars = args.arg_map;

    // Print usage
    printf("%s\n", help);

    // Read user specified args
    string fmap = "";
    if (pars.count("in") && !pars["in"].empty())
        fmap = pars["in"][0];
    if (fmap == ""){
        printf("# Please input map\n");
        exit(1);
    }
    printf("# Setting input map directory to %s\n", fmap.c_str());

    string fout = "./points.pdb";
    if (pars.count("out") && !pars["out"].empty())
        fout = pars["out"][0];
    printf("# Setting output points directory to %s\n", fout.c_str());

    float thresh = 5.0;
    if (pars.count("thresh") && !pars["thresh"].empty())
        thresh = std::stod(pars["thresh"][0]);
    printf("# Setting map thresh to %.4f\n", thresh);

    float resol = 6.0;
    if (pars.count("res") && !pars["res"].empty())
        resol = std::stod(pars["res"][0]);
    printf("# Setting map resolution to %.4f\n", resol);
   
    int nt = 4;
    if (pars.count("nt") && !pars["nt"].empty())
        nt = std::stoi(pars["nt"][0]);
    printf("# Setting num threads to %d\n", nt);

    float rmax = 10.0;
    if (pars.count("rmax") && !pars["rmax"].empty())
        rmax = std::stod(pars["rmax"][0]);
    printf("# Setting max shift to %.4f\n", rmax);

    // Read map
    MRC* mrc = new MRC;
    read_mrc(fmap.c_str(), mrc);
    flatten(mrc);

    // Thresh
    printf("# Thresh map\n");
    thresh_map(mrc, thresh);

    // Lower-bound filter
    float filter = 0.01;
    if (pars.count("filter") && !pars["filter"].empty())
        filter = std::stod(pars["filter"][0]);
    printf("# Setting low-bound filter to %.4f\n", filter);

    // Setting merge distance
    float dmerge = 1.0;
    if (pars.count("dmerge") && !pars["dmerge"].empty())
        dmerge = std::stod(pars["dmerge"][0]);
    printf("# Setting dmerge to %.4f\n", dmerge);

    // If providing initial starting points
    std::vector<std::vector<float>> init_crds;
    if (pars.count("pdb") && !pars["pdb"].empty()){
        std::string fpdb = pars["pdb"][0];
        printf("# Setting initial shift location to %s\n", fpdb.c_str());
        read_pdb(fpdb, init_crds);
    }
    printf("# Get %d initial locations\n", init_crds.size());

    // MS
    omp_set_num_threads(nt);
    printf("# Converting maps to local maximums, running on %d threads\n", nt);

    Points* pp = new Points;
    msshift(mrc, pp, resol, rmax, init_crds);
    msmerge(mrc, pp, filter, dmerge);
  
    // Read map again
    MRC* mrc0 = new MRC;
    read_mrc(fmap.c_str(), mrc0);
    flatten(mrc0);
    thresh_and_norm_map(mrc0, 1e-6);    

    // Get density
    //vector<double> density;
    //for(int i = 0; i < pp->coords.size(); ++i){
    //    double dens = get_density(mrc0, pp->coords[i]);
    //    density.push_back(dens);
    //}
    vector<double> density(pp->coords.size(), 0.0);
    double dmax =  0.0;
    double dmin =  1e9;
    for(int i = 0; i < pp->coords.size(); ++i){
        double dens = pp->dens[i];
        dmax = std::max(dens, dmax);
        dmin = std::min(dens, dmin);
    }
    double drange = dmax - dmin;
    for(int i = 0; i < pp->coords.size(); ++i){
        double dens = pp->dens[i];
        density[i] = (dens - dmin) / (1e-6 + drange);
    }

    // Output 
    write_points_and_bfactors_as_pdb(pp->coords, density, fout, true);

    return 0;
}
