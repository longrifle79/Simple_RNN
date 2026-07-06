#include "randomize_weight.h"

Randomize_Weight::Randomize_Weight()
{
    
}

Randomize_Weight::Randomize_Weight(float min, float max)
{
    minimum = min;
    maximum = max;
}


float Randomize_Weight::getFloat(float min, float max) 
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

float Randomize_Weight::getFloat() 
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minimum, maximum);
    return dist(gen);
}

