#ifndef __INPUT_GATE_H__
#define __INPUT_GATE_H__
#include "activation_functions.h"

class Input_Gate : Activation_Functions
{
    public:
        Input_Gate() = default;

        float compute_input_gate(   float input, 
                                    float input_weight1, 
                                    float input_weight2,
                                    float short_term_memory, 
                                    float short_term_memory_weight1, 
                                    float short_term_memory_weight2, 
                                    float input_bias1,
                                    float input_bias2);
};


#endif // __INPUT_GATE_H__

