#include "forget_gate.h"


float ForgetGate::compute_forget_gate(  float input, 
                                        float input_weight,
                                        float prev_shrt_trm_mem, 
                                        float prev_shrt_trm_mem_weight, 
                                        float forget_bias)
{
    float sum = (input * input_weight) + 
                (prev_shrt_trm_mem * prev_shrt_trm_mem_weight) + 
                forget_bias;
                
    return sigmoid(sum);
};