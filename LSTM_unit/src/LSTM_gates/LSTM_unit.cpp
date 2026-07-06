#include "lstm_unit.h"




        int Lstm::compute_lstm_unit(float input, 
                                    float short_term_memory, 
                                    float long_term_memory, 
                                    LSTM_unit_weights lstm_unit_weights,
                                    float &new_short_term_memory,
                                    float &new_long_term_memory)
        {
            // float forget_gate_output = forget_gate.compute_forget_gate( input, 
            //                                                             lstm_unit_weights.get_forget_gate_wxf(), 
            //                                                             short_term_memory, 
            //                                                             lstm_unit_weights.get_forget_gate_whf(), 
            //                                                             lstm_unit_weights.get_forget_gate_bf());
        }


        // {
        //     float forget_gate_output = forget_gate.compute_forget_gate(input, 
        //                                                                 lstm_unit_weights.input_weight1, 
        //                                                                 short_term_memory, 
        //                                                                 lstm_unit_weights.short_term_memory_weight1, 
        //                                                                 lstm_unit_weights.input_bias1);

        //     float input_gate_output = input_gate.compute_input_gate(input, 
        //                                                             lstm_unit_weights.input_weight1, 
        //                                                             lstm_unit_weights.input_weight2,
        //                                                             short_term_memory, 
        //                                                             lstm_unit_weights.short_term_memory_weight1, 
        //                                                             lstm_unit_weights.short_term_memory_weight2, 
        //                                                             lstm_unit_weights.input_bias1,
        //                                                             lstm_unit_weights.input_bias2);

        //     float output_gate_output = output_gate.compute_output_gate(input, 
        //                                                                 lstm_unit_weights.input_weight1, 
        //                                                                 short_term_memory, 
        //                                                                 lstm_unit_weights.short_term_memory_weight1, 
        //                                                                 lstm_unit_weights.input_bias1,
        //                                                                 long_term_memory);

        //     long_term_memory = forget_gate_output * long_term_memory + input_gate_output;
        //     short_term_memory = output_gate_output * tanh(long_term_memory);

        //     return short_term_memory;
        // }








