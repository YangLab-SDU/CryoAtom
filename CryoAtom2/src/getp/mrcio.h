#ifndef MRCIO_H
#define MRCIO_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <exception>
#include <cassert>

class MRC{
public:
    int ncrs[3]; 
    int mode;
    int ncrsstart[3];
    int mxyz[3];
    float cella[3];
    float cellb[3];
    int mapcrs[3];
    float dmin, dmax, dmean;
    int ispg;
    int nsymbt;
    int extra[25];
    float originxyz[3];
    char map[4];
    char machst[4];
    float rms;
    int nlabel;
    char label[10][80];
    
    char* sym;

    float*** cube;
    
    float* line;
    int Nvoxel;
    int Nact;
    float vsize[3];
    float ORIGIN[3];
    MRC() : line(NULL), cube(NULL) { }

    // Deconstructor
    ~MRC() {
        if(line != NULL)
            delete[] line;
    }
};

int get_mode(int* mapcrs);
int get_index(int mode, int count, int nc, int nr, int ns);
bool check_orthogonal(float* cellb);
void read_mrc(const char* file, MRC* mrc);
void write_mrc(const char* file, MRC* mrc);
void flatten(MRC* mrc);
void to_cubic(MRC* mrc);

#endif
