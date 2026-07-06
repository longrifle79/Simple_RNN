#include "input_gate.h"


        float Input_Gate::compute_input_gate(    float input, 
                                                float input_weight1, 
                                                float input_weight2,
                                                float short_term_memory, 
                                                float short_term_memory_weight1, 
                                                float short_term_memory_weight2, 
                                                float input_bias1,
                                                float input_bias2)
        {
            float scaled_pot_mem = input * input_weight1 + short_term_memory * short_term_memory_weight1 + input_bias1;
            scaled_pot_mem = sigmoid(scaled_pot_mem);

            float pot_long_term_mem = input * input_weight2 + short_term_memory * short_term_memory_weight2 + input_bias2;
            pot_long_term_mem = tanh(pot_long_term_mem);

            return scaled_pot_mem * pot_long_term_mem;
        }