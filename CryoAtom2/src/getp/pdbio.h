#ifndef PDBIO_H
#define PDBIO_H

#include <cstdio>
#include <cstdlib>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <utility>
#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::set;
using std::map;
using std::pair;

#include <exception>
#include <stdexcept>

void read_pdb(string& filename, vector<vector<float>>& crds);

//Write points with bfactors as pdb
template<typename Arr2D, typename Arr1D>
void write_points_and_bfactors_as_pdb(Arr2D& p, Arr1D& bfactors, string filename, bool write_ter){
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "# Failed to open file " << filename << std::endl;
        return;
    }
    if (p.size() != bfactors.size() ){
        std::cerr << "# Num of points (" << p.size() << ") dont match num of bfactors (" << bfactors.size() << ")" << std::endl;
        exit(1);
    }

    for (int i = 0; i < p.size(); ++i) {
        // Update 2024-05-13
        // Do not show atom number and res number when > 9999
        int count = i + 1;
        if (count > 9999)
            count = 9999;

        outFile << "ATOM  " << std::setw(5) << count << "  CA  GLY A" << std::setw(4) << count
                << "    "
                << std::right
                << std::fixed
                << std::setprecision(3)
                << std::setw(8) << p[i][0]
                << std::setw(8) << p[i][1]
                << std::setw(8) << p[i][2] 
                << std::setprecision(2)
                << std::setw(6) << bfactors[i]
                << std::setw(6) << bfactors[i]
                << "              "
                << std::endl;

        if (write_ter)
            outFile << "TER" << endl;
    }
    if (!write_ter)
        outFile << "TER" << endl;
    //outFile << "END" << endl;
    outFile.close();
}

#endif
