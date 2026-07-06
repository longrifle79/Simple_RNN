#include "lstm_unit.h"


        // Forget_Gate forgetGate;
        // Input_Gate inputGate;
        // Output_Gate outputGate;
        // Lstm_Unit_Weights lstmUnitWeights;



        int Lstm::compute_lstm_unit(float input, 
                                    float short_term_memory, 
                                    float long_term_memory, 
                                    Lstm_Unit_Weights lstmUnitWeights,
                                    float &new_short_term_memory,
                                    float &new_long_term_memory)
        {
            float forget_gate_output = forgetGate.compute_forget_gate( input, 
                                                                        lstmUnitWeights.get_forget_gate_wxf(), 
                                                                        short_term_memory, 
                                                                        lstmUnitWeights.get_forget_gate_whf(), 
                                                                        lstmUnitWeights.get_forget_gate_bf());

            float scaled_long_term_memory = long_term_memory * forget_gate_output;

            float input_gate_output = inputGate.compute_input_gate( input, 
                                                                    lstmUnitWeights.get_input_scaled_potential_memory_wxf(), 
                                                                    lstmUnitWeights.get_input_potential_memory_wxf(),
                                                                    short_term_memory,
                                                                    lstmUnitWeights.get_input_scaled_potential_memory_whf(),
                                                                    lstmUnitWeights.get_input_potential_memory_whf(),
                                                                    lstmUnitWeights.get_input_scaled_potential_memory_bf(),
                                                                    lstmUnitWeights.get_input_potential_memory_bf());

            float computed_long_term_memory = scaled_long_term_memory * input_gate_output;

            float computed_short_term_memory = outputGate.compute_output_gate(   input, 
                                                                            lstmUnitWeights.get_output_scaled_potential_memory_wxf(),
                                                                            short_term_memory,
                                                                            lstmUnitWeights.get_output_scaled_potential_memory_whf(),
                                                                            lstmUnitWeights.get_output_scaled_potential_memory_bf(),
                                                                            computed_long_term_memory);

            new_long_term_memory = computed_long_term_memory;
            new_short_term_memory = computed_short_term_memory;

            return (1);


        }


 






