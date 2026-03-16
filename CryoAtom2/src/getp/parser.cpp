#include "parser.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <deque>
#include <map>
using std::string;
using std::cout;
using std::endl;
using std::map;
using std::deque;
#include <algorithm>



// Convert a string to lower case
string to_lower(string input) {
    string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower;
}

// Remove continuous prefix in input
string remove_prefix(string input, char prefix) {
    if(input.empty())
        return "";
    int k = 0;
    while(k < input.size() && input[k] == prefix)
        ++k;
    if(k >= input.size())
        return "";
    else
        return input.substr(k);
}


Args parse_args(int argc, char* argv[]){
    Args args;
    std::map<std::string, std::string> arg_map;

    string last = "others";

    for(int i = 1; i < argc; ++i){
        string curr(argv[i]);

        if (curr.size() >= 2 && curr[0] == '-' && curr[1] == '-') {
            // If startswith '--', it is an option
            curr = remove_prefix(curr, '-');
            if (curr != last){
                // Create a empty arr
                last = curr;
                args.arg_map[last];
            }
        }else{
            // If not, it is a param
            args.arg_map[last].push_back(curr);
        }
        
    }
    return args;
}


// Print all options
void print_options(bool detail){
    printf("# See http://huanglab.phys.hust.edu.cn/EMRNA/ for details\n");
}
