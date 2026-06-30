#ifndef __FORGET_GATE_H__
#define __FORGET_GATE_H__
#include "activation_functions.h"


class ForgetGate: ActivationFunctions
{
    public:
        ForgetGate() = default;

        float compute_forget_gate(float input, float input_weight,float prev_shrt_trm_mem, float prev_shrt_trm_mem_wieght, float forget_bias);
};





#endif // __FORGET_GATE_H__

