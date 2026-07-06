#ifndef __OUTPUT_GATE_H__
#define __OUTPUT_GATE_H__
#include "activation_functions.h"

class Output_Gate : Activation_Functions
{
    public:
        Output_Gate() = default;

        float compute_output_gate(  float input, 
                                    float input_weight1, 
                                    float short_term_memory, 
                                    float short_term_memory_weight1, 
                                    float input_bias1,
                                    float cell_state);
};



#endif // __OUTPUT_GATE_H__

