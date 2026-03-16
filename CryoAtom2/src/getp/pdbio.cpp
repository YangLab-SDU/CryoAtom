#include "pdbio.h"

#include <fstream>
#include <string>
#include <vector>
using std::ifstream;
using std::string;
using std::vector;

void read_pdb(string& filename, vector<vector<float>>& crds){
    ifstream fin(filename, std::ios::in);
    string line;
    while(std::getline(fin, line)){
        if (line.size() >= 54 && line.substr(0, 4) == "ATOM"){
            float x = std::stod(line.substr(30, 8));
            float y = std::stod(line.substr(38, 8));
            float z = std::stod(line.substr(46, 8));

            crds.push_back({x, y, z});
        }
    }
}
