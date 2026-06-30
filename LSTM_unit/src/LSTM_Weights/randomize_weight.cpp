#include "randomize_weight.h"

RandomizeWeight::RandomizeWeight()
{
    // Constructor implementation (if needed)
}

RandomizeWeight::RandomizeWeight(float min, float max)
{
    minimum = min;
    maximum = max;
}


float RandomizeWeight::getFloat(float min, float max) 
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

float RandomizeWeight::getFloat() 
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minimum, maximum);
    return dist(gen);
}

