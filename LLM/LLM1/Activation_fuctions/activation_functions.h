#ifndef __ACTIVATION_FUNCTIONS_H__
#define __ACTIVATION_FUNCTIONS_H__
#include "sigmoid.h"
#include "tanh.h"


class Activation_Functions : public Tanh, public Sigmoid
{
public:
    Activation_Functions() = default;

    using Tanh::tanh;
    using Tanh::tanh_derivative;

    using Sigmoid::sigmoid;
    using Sigmoid::sigmoid_derivative;
};

#endif // __ACTIVATION_FUNCTIONS_H__