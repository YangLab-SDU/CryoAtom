#ifndef PARSER_H
#define PARSER_H

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


class Args{
public:
    map<string, deque<string>> arg_map;
};


// Convert a string to lower case
string to_lower(string input);

// Remove continuous prefix in input
string remove_prefix(string input, char prefix);

// Parse arguments
Args parse_args(int argc, char* argv[]);

// Print all options
void print_options(bool detail = false);

#endif
