#ifndef __FORGET_GATE_H__
#define __FORGET_GATE_H__
#include "activation_functions.h"


class Forget_Gate : Activation_Functions
{
    public:
        Forget_Gate() = default;

        float compute_forget_gate(  float input, 
                                    float input_weight, 
                                    float short_term_memory, 
                                    float short_term_memory_weight, 
                                    float forget_bias);


};





#endif // __FORGET_GATE_H__

