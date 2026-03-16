#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include "mrcio.h"

using std::vector;

class Points{
public:
    double **cd,*dens;
    int **origrid;
    int Ncd,Nori;
    int *member;
    double *mask;
    bool *ignore;

    int Np;
    double dmax;
    double dmin;

    vector<vector<double>> coords;
    vector<vector<double>> cds;
    vector<double> centroid;
    
    Points() : Np(0), Ncd(0), Nori(0), dmax(1.0), dmin(0.0) { }
};



#ifndef PI
    #define PI 3.1415926535898
#endif


bool msshift(MRC *m, Points *p, float res, float max_shift, vector<vector<float>>& init_crds);
bool msmerge(MRC *m, Points *p, double filter, double distance);


#endif
