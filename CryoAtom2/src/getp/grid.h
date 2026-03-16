#ifndef GRID_H
#define GRID_H


#include <vector>
#include <cmath>
#include <algorithm>
using std::vector;

#include "mrcio.h"

void min_max_norm(vector<double>& dens);

void min_max_norm_with_factor(vector<double>& dens, vector<double>& mins, vector<double>& maxs);

//EM get density
vector<double> em_get_density(MRC* map, vector<vector<double>>& coords);
double get_density(MRC* map, vector<double>& coord);


//Get density by coord
template<typename Coord>
double em_get_density(MRC* map, Coord& coord);


//EM get aa types
vector<int> em_get_aa(MRC* map, vector<vector<double>>& coords, const int& n_classes = 4);


template<typename Coord>
int em_get_aa_coord(MRC* map, Coord& coord, const int& n_classes);

void thresh_map(MRC* map, float th);
void norm_map(MRC* map);
void thresh_and_norm_map(MRC* map, float th);


#endif
