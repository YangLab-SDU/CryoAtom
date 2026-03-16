#include "grid.h"

#include <vector>
#include <cmath>
#include <algorithm>
using std::vector;

void min_max_norm(vector<double>& dens){
    double dmax = -1e9;
    double dmin =  1e9;
    for(int i = 0; i < dens.size(); ++i){
        dmax = std::max(dmax, dens[i]);
        dmin = std::min(dmin, dens[i]);
    }
    for(int i = 0; i < dens.size(); ++i)
        dens[i] = (dens[i] - dmin) / (dmax - dmin);
}

void min_max_norm_with_factor(vector<double>& dens, vector<double>& mins, vector<double>& maxs){
    for(int i = 0; i < dens.size(); ++i){
        double dmax = maxs[i];
        double dmin = mins[i];
        dens[i] = (dens[i] - dmin) / (dmax - dmin);
    }
}

//EM get density
vector<double> em_get_density(MRC* map, vector<vector<double>>& coords){
    vector<double> dens(coords.size(), 0.0);
    for(int i = 0; i < coords.size(); ++i){
        dens[i] = em_get_density(map, coords[i]);
    }
    //Norm
    min_max_norm(dens);
    return dens;
}

double get_density(MRC* map, vector<double>& coord){
    int tempx = (coord[0] - map->ORIGIN[0]) / map->vsize[0];
    int tempy = (coord[1] - map->ORIGIN[1]) / map->vsize[1];
    int tempz = (coord[2] - map->ORIGIN[2]) / map->vsize[2];

    if( tempx < 0 || tempx >= map->mxyz[0] ||
        tempy < 0 || tempy >= map->mxyz[1] ||
        tempz < 0 || tempz >= map->mxyz[2] )
        return 0.0;
    else{
        int index = tempz * map->mxyz[1] * map->mxyz[0] + tempy * map->mxyz[0] + tempx;
        return map->line[index];
    }
}



//Get density by coord
template<typename Coord>
double em_get_density(MRC* map, Coord& coord){

    double tempx = coord[0] - map->ORIGIN[0];
    double tempy = coord[1] - map->ORIGIN[1];
    double tempz = coord[2] - map->ORIGIN[2];

    int kx0=int(tempx / map->vsize[0]);
    int kx1=kx0+1;
    int ky0=int(tempy / map->vsize[1]);
    int ky1=ky0+1;
    int kz0=int(tempz / map->vsize[2]);
    int kz1=kz0+1;

    if( kx0 < 0 || kx0 >= map->mxyz[0] ||
        kx1 < 0 || kx1 >= map->mxyz[0] ||
        ky0 < 0 || ky0 >= map->mxyz[1] ||
        ky1 < 0 || ky1 >= map->mxyz[1] ||
        kz0 < 0 || kz0 >= map->mxyz[2] ||
        kz1 < 0 || kz1 >= map->mxyz[2] )
        return 0.0;

    double x0=tempx/map->vsize[0] - kx0;
    double y0=tempy/map->vsize[1] - ky0;
    double z0=tempz/map->vsize[2] - kz0;

    double txy=x0*y0;
    double tyz=y0*z0;
    double txz=x0*z0;
    double txyz=x0*y0*z0;

    int index000 = kz0 * map->mxyz[1] * map->mxyz[0] + ky0 * map->mxyz[0] + kx0;
    int index100 = kz1 * map->mxyz[1] * map->mxyz[0] + ky0 * map->mxyz[0] + kx0;
    int index010 = kz0 * map->mxyz[1] * map->mxyz[0] + ky1 * map->mxyz[0] + kx0;
    int index001 = kz0 * map->mxyz[1] * map->mxyz[0] + ky0 * map->mxyz[0] + kx1;
    int index101 = kz1 * map->mxyz[1] * map->mxyz[0] + ky0 * map->mxyz[0] + kx1;
    int index011 = kz0 * map->mxyz[1] * map->mxyz[0] + ky1 * map->mxyz[0] + kx1;
    int index110 = kz1 * map->mxyz[1] * map->mxyz[0] + ky1 * map->mxyz[0] + kx0;
    int index111 = kz1 * map->mxyz[1] * map->mxyz[0] + ky1 * map->mxyz[0] + kx1;

    double v000=map->line[index000];
    double v100=map->line[index100];
    double v010=map->line[index010];
    double v001=map->line[index001];
    double v101=map->line[index101];
    double v011=map->line[index011];
    double v110=map->line[index110];
    double v111=map->line[index111];

    double temp1=v000*(1.0-x0-y0-z0+txy+tyz+txz-txyz);
    double temp2=v100*(x0-txy-txz+txyz);
    double temp3=v010*(y0-txy-tyz+txyz);
    double temp4=v001*(z0-txz-tyz+txyz);
    double temp5=v101*(txz-txyz);
    double temp6=v011*(tyz-txyz);
    double temp7=v110*(txy-txyz);
    double temp8=v111*txyz;

    return temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8;
}



//EM get aa types
vector<int> em_get_aa(MRC* map, vector<vector<double>>& coords, const int& n_classes){
    vector<int> aa(coords.size(), 4);
    for(int i = 0; i < coords.size(); ++i){
        aa[i] = em_get_aa_coord(map, coords[i], n_classes);
    }
    return aa;
}


template<typename Coord>
int em_get_aa_coord(MRC* map, Coord& coord, const int& n_classes){
    int tempx = coord[0] - map->ORIGIN[0];
    int tempy = coord[1] - map->ORIGIN[1];
    int tempz = coord[2] - map->ORIGIN[2];

    int kx0=int(tempx / map->vsize[0]);
    int xl = kx0 - 1, xh = kx0 + 1;
    int kx1=kx0 + 1;
    int ky0=int(tempy / map->vsize[1]);
    int yl = ky0 - 1, yh = ky0 + 1;
    int ky1=ky0 + 1;
    int kz0=int(tempz / map->vsize[2]);
    int zl = kz0 - 1, zh = kz0 + 1;
    int kz1=kz0 + 1;

    if( kx0 < 0 || kx0 >= map->mxyz[0] ||
        kx1 < 0 || kx1 >= map->mxyz[0] ||
        ky0 < 0 || ky0 >= map->mxyz[1] ||
        ky1 < 0 || ky1 >= map->mxyz[1] ||
        kz0 < 0 || kz0 >= map->mxyz[2] ||
        kz1 < 0 || kz1 >= map->mxyz[2] )
        return 4;

    //Vote for 27 voxels
    vector<int> counts(5, 0);
    for(int xx = xl; xx <= xh; ++xx){
        if(xx < 0 || xx >= map->mxyz[0])
            continue;
        for(int yy = yl; yy <= yh; ++yy){
            if(yy < 0 || yy >= map->mxyz[1])
                continue;
            for(int zz = zl; zz <= zh; ++zz){
                if(zz < 0 || zz >= map->mxyz[2])
                    continue;

                int index = zz * map->mxyz[1] * map->mxyz[0] + yy * map->mxyz[0] + xx;
                int v = int(map->line[index]);

                // 10-30 AGCU 0123
                /*
                if(n_classes == 4)
                    // 01234 -> 40123
                    v = (v + 4) % 5;
                else if(n_classes == 2){
                    if(v == 0)
                        v = 4;
                    if(v == 1)
                        v = 0;
                }
                */

                ++counts[v];
            }
        }
    }
    auto it = std::max_element(counts.begin(), counts.end());
    return int(it - counts.begin());
}


// Thresh map
void thresh_map(MRC* map, float th){
    for(int i = 0; i < map->Nvoxel; ++i){
        if(map->line[i] < th)
            map->line[i] = 0.0f;
    }
}

// Norm map
void norm_map(MRC* map){
    float dmax = -1e6;
    float dmin =  1e6;
    for(int i = 0; i < map->Nvoxel; ++i){
        dmax = std::max(dmax, map->line[i]);
        dmin = std::min(dmin, map->line[i]);
    }
    for(int i = 0; i < map->Nvoxel; ++i){
        map->line[i] = (map->line[i] - dmin) / (dmax - dmin);
    }
}


void thresh_and_norm_map(MRC* map, float th){
    thresh_map(map, th);
    norm_map(map);
}
