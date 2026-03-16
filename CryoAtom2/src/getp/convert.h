#ifndef CONVERT_H
#define CONVERT_H

// For eigen
#include <Eigen/Dense>
using Mat = Eigen::MatrixXd;

#include <set>
#include <string>
#include <vector>
using std::set;
using std::vector;
using std::string;

#include "pdbio.h"


// Convert between different types
// 1. Eigen::MatrixXd and std::vector<vector<double>>
Mat vector2mat(vector<vector<double>>& v);
vector<vector<double>> mat2vector(Mat& m);

// 2. Chain and vector<vector<double>>
vector<vector<double>> chain2vector(Chain& c, string atom = "C4'");
Chain vector2chain(vector<vector<double>>& c, string atom = "C4'");

// 3. Chain and mat - using chain2vector, and vector2mat and vice versa.
Mat chain2mat(Chain& );

// 4. Residue and mat
Mat residue2mat(const Residue& res, const std::set<std::string>& selection);

// Convert
void convert_atom_name(Chain& chain, char from, char to);


#endif
