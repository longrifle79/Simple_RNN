#include "lsr.h"



float y_value(const std::list<std::list<float>>& data, int row)
{
    // Declare a const iterator for the outer list and initialize to the first row.
    // Same as before: read-only access.
    std::list<std::list<float>>::const_iterator row_it = data.begin();

    // Advance to the desired row (0 based index).
    // Unchanged: O(row) time.
    std::advance(row_it, row);  // Move iterator to the desired row index

    // Get an iterator to the first element (column 0, the x value) in the selected row.
    // Unchanged.
    std::list<float>::const_iterator col_it = row_it->begin();

    // Hardcode advance by exactly 1 step.
    // This skips column 0 (x) and lands on column 1 (y).
    // Always moves to the second element, assuming it exists.
    std::advance(col_it, 1);  // Move iterator to the desired column index (fixed to y-value)

    // Dereference and return the y value.
    // Unchanged.
    return *col_it;
}

float x_value(const std::list<std::list<float>>& data, int row)
{
        // Declare a const iterator for the outer list and initialize to the first row.
    // Same as before: read-only access.
    std::list<std::list<float>>::const_iterator row_it = data.begin();

    // Advance to the desired row (0 based index).
    // Unchanged: O(row) time.
    std::advance(row_it, row);  // Move iterator to the desired row index

    // Get an iterator to the first element (column 0, the x value) in the selected row.
    // Unchanged.
    std::list<float>::const_iterator col_it = row_it->begin();

    // Hardcode advance by exactly 1 step.
    // This skips column 0 (x) and lands on column 1 (y).
    // Always moves to the second element, assuming it exists.
    std::advance(col_it, 0);  // Move iterator to the desired column index (fixed to y-value)

    // Dereference and return the y value.
    // Unchanged.
    return *col_it;
}

float list_average_y(const std::list<std::list<float>> list)
{
    std::list<std::list<float>>::const_iterator row_it = list.begin();
    std::advance(row_it, 0);  // Move iterator to the desired row index
    float sum = 0.0;

    for(int i = 0; i < list.size(); i++)
    {
        sum += y_value(list, i);
    }
    return sum / list.size();
}

float list_average_x(const std::list<std::list<float>> list)
{
        std::list<std::list<float>>::const_iterator row_it = list.begin();
    std::advance(row_it, 0);  // Move iterator to the desired row index
    float sum = 0.0;

    for(int i = 0; i < list.size(); i++)
    {
        sum += x_value(list, i);
    }
    return sum / list.size();
}

float ssr(const std::list<std::list<float>>& data, float slope, float y_intercept)
{
    float ssr_sum = 0.0;
    for(int i = 0; i < data.size(); i++)
    {
        float x = x_value(data, i);
        float y = y_value(data, i);

        float predicted_y = ((slope * x) + y_intercept);

        ssr_sum += (y - predicted_y) * (y - predicted_y);
    }
    return ssr_sum;
}

float least_squares(std::list<std::list<float>> data, float learning_rate)
{
    float degrees = 0.0;
    float current_slope = 0.0;
    float best_slope = std::numeric_limits<float>::max();
    float current_ssr_slope;
    float best_ssr_slope = std::numeric_limits<float>::max();

    while(degrees <= 180)
    {
        std::cout << "Calculating Slope:  " << best_slope << std::endl;

        current_slope = degree_to_slope(degrees);
        current_ssr_slope = ssr(data, current_slope, 0);

        if(current_ssr_slope < best_ssr_slope)
        {
            best_slope = current_slope;
            best_ssr_slope = current_ssr_slope;
        }
        degrees += learning_rate;
    }

    int high_counter = 0;
    float current_ssr = 0.0;
    float best_ssr= std::numeric_limits<float>::max();
    float yint1 = 0.0;

    while(high_counter < MAX_DIRECTION)
    {
        std::cout << "Calculating y intercept first direction" << yint1;
        std::cout << "     High Counter: " << high_counter << std::endl;

        current_ssr = ssr(data, best_slope, yint1);

        if(current_ssr < best_ssr)
        {
            best_ssr = current_ssr;
            high_counter = 0;
        }
        else 
        {
            high_counter++;
        }
        yint1 += learning_rate;
    }

    high_counter = 0;
    float yint2 = 0.0;

    while(high_counter < MAX_DIRECTION)
    {
        std::cout << "Calculating y intercept second direction" << yint2;
        std::cout << "     High Counter: " << high_counter << std::endl;

        current_ssr = ssr(data, best_slope, yint2);

        if(current_ssr < best_ssr)
        {
            best_ssr = current_ssr;
            high_counter = 0;
        }
        else 
        {
            high_counter++;
        }
        yint2 -= learning_rate;
    }

    float ssrTest1 = ssr(data, best_slope, yint1);
    float ssrTest2 = ssr(data, best_slope, yint2);

    if(ssrTest1 < ssrTest2)
    {
        std::cout << "Slope:  " << best_slope << "      y intercept:  " << yint1 << std::endl;
        return ssrTest1;        
    }
    else
    {
        std::cout << "Slope:  " << best_slope << "      y intercept:  " << yint2 << std::endl;
        return ssrTest2;
    }
}



































float degree_to_slope(float angle_deg)
{
    if(angle_deg == 90)
    {
        angle_deg = 89.999999;
        std::cout << "***WARNING SLOPE IS AT 90 degrees NOT A REAL NUMBER***" << std::endl;
    }
    if(angle_deg == 270)
    {
        angle_deg = 269.9999999;
        std::cout << "***WARNING SLOPE IS AT 270 degrees NOT A REAL NUMBER***" << std::endl;
    }

    double angle_radians = static_cast<double>(angle_deg) * PI / 180;
    double slope_double = std::tan(angle_radians);

    if(slope_double > std::numeric_limits<double>::max())
    {
        slope_double = std::numeric_limits<double>::max();
        std::cout << "***WARNING SLOPE TOO LARGE! ADJUSTED TO FIT DATA***" << std::endl;
    }

    return static_cast<float>(slope_double);
}

