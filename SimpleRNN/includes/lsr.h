#ifndef __LSR_H__
#define __LSR_H__

#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cstdlib>  // for strtof, EXIT_FAILURE
#include <cstring>  // for memset
#include <filesystem> // for std::filesystem::exists
#include "csvTools.h"   // for csv2list, printCSV, load_all_csv_from_directory, printCSV_all
#include <cmath>      // For std::tan and M_PI (with define)
#include <limits>     // For std::numeric_limits (optional NaN handling)


#define MAX_SLOPE   1000000000
#define PI          3.14159265359
#define MAX_DIRECTION   10

#define LEARNING_RATE   0.001

float y_value(const std::list<std::list<float>>& data, int row, int col);
float x_value(const std::list<std::list<float>>& data, int row, int col);

float list_average_y(const std::list<std::list<float>> list);
float list_average_x(const std::list<std::list<float>> list);

float ssr(const std::list<std::list<float>>& data, float slope, float y_intercept);

float least_squares(std::list<std::list<float>> data, float learning_rate);

float degree_to_slope(float);




#endif // __LSR_H__

