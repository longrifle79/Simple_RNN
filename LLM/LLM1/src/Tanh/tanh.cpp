#include "tanh.h"



float Tanh::tanh(float x) 
{
    return std::tanh(x);
}



float Tanh::tanh_derivative(float x) 
{
    float t = tanh(x);
    return 1.0f - t * t;
}



