#ifndef __SIGMOID_H__
#define __SIGMOID_H__
#include <cmath>

class Sigmoid 
{
public:

    // Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    float sigmoid(float x);

    // Derivative of sigmoid (very useful for backpropagation)
    float sigmoid_derivative(float x);
};


#endif

