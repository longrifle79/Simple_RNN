#ifndef __ACTIVATION_FUNCTIONS_H__
#define __ACTIVATION_FUNCTIONS_H__
#include "sigmoid.h"
#include "tanh.h"


class ActivationFunctions : public Tanh, public Sigmoid
{
public:
    ActivationFunctions() = default;

    // Bring methods from base classes into this scope so we can call them cleanly
    using Tanh::tanh;
    using Tanh::tanh_derivative;

    using Sigmoid::sigmoid;
    using Sigmoid::sigmoid_derivative;
};

#endif // __ACTIVATION_FUNCTIONS_H__