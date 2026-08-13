#include "sigmoid.h"


float Sigmoid::sigmoid(float x) 
{
    return 1.0f / (1.0f + std::exp(-x));
}

float Sigmoid::sigmoid_derivative(float x) 
{
    float s = sigmoid(x);
    return s * (1.0f - s);
}