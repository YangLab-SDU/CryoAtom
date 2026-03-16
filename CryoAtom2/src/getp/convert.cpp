#include "convert.h"

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
Mat vector2mat(vector<vector<double>>& v){
    int rows = v.size();
    int cols = v[0].size();
    Mat m(rows, cols);
    for(int i = 0; i < v.size(); ++i){
        for(int j = 0; j < v[0].size(); ++j)
            m(i, j) = v[i][j];
    }
    return m;
}

vector<vector<double>> mat2vector(Mat& m){
    int rows = m.rows();
    int cols = m.cols();
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = m(i, j);
        }
    }
    return result;
}


// 2. Chain and vector<vector<double>>
vector<vector<double>> chain2vector(Chain& c, string atom){
    vector<vector<double>> ret(c.size(), vector<double>(3, 0.0));
    for(int i = 0; i < c.size(); ++i){
        for(int k = 0; k < 3; ++k)
            ret[i][k] = c[i][atom][k];
    }
    return ret;
}



// 3. Chain and mat - using chain2vector, and vector2mat and vice versa.
Mat chain2mat(Chain& chain){
    auto vec = chain2vector(chain);
    return vector2mat(vec);
}


// 4. Residue and mat
Mat residue2mat(const Residue& res, const std::set<string>& selection){
    Mat mat(selection.size(), 3);
    int n = 0;
    for(auto& choice : selection){
        for(auto& atom : res){
            if(atom.name == choice){
                for(int k = 0; k < 3; ++k)
                    mat(n, k) = atom[k];
                ++n;
                break;
            }
        }
    }
    return mat;
}


// Convert atom name
void convert_atom_name(Chain& chain, char from, char to){
    for(auto&& res : chain){
        for(auto&& atom : res){
            auto it = std::find(atom.name.begin(), atom.name.end(), from);
            if (it != atom.name.end()){
                *it = to;
            }
        }
    }
}

