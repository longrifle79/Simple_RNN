#include "output_gate.h"


        float Output_Gate::compute_output_gate(  float input, 
                                                float input_weight1, 
                                                float short_term_memory, 
                                                float short_term_memory_weight1, 
                                                float input_bias1,
                                                float cell_state)
        {
            float scaled_pot_mem = input * input_weight1 + short_term_memory * short_term_memory_weight1 + input_bias1;
            scaled_pot_mem = sigmoid(scaled_pot_mem);

            float pot_long_term_mem = cell_state;
            pot_long_term_mem = tanh(pot_long_term_mem);

            return scaled_pot_mem * pot_long_term_mem;
        }