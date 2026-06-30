#include "LSTM_unit_weights_and_bias.h"



        LSTM_unit_weights::LSTM_unit_weights()
        {
                // Initialize Forget Gate
                LSTM_weights.push_back(section_weight_and_bias());


                // Initialize Input Gate
                // Initialize Input Scaled Potential Memory
                LSTM_weights.push_back(section_weight_and_bias()); 
                // Initialize Input Potential Memory
                LSTM_weights.push_back(section_weight_and_bias()); 


                // Initialize Output Gate
                LSTM_weights.push_back(section_weight_and_bias()); 
                // Initialize Output Scaled Potential Memory
                LSTM_weights.push_back(section_weight_and_bias());
        }


//*****************   Forget Gate Weights and Biases   *****************//
        float LSTM_unit_weights::get_forget_gate_wxf()
        {
                return LSTM_weights[0].get_wx();
        }
        int LSTM_unit_weights::set_forget_gate_wxf(float new_weight)
        {
                return LSTM_weights[0].set_wx(new_weight);
        }

        float LSTM_unit_weights::get_forget_gate_whf()
        {
                return LSTM_weights[0].get_wh();
        }
        int LSTM_unit_weights::set_forget_gate_whf(float new_weight)
        {
                return LSTM_weights[0].set_wh(new_weight);
        }

        float LSTM_unit_weights::get_forget_gate_bf()
        {
                return LSTM_weights[0].get_b();
        }
        int LSTM_unit_weights::set_forget_gate_bf(float new_bias)
        {
                return LSTM_weights[0].set_b(new_bias);
        }


//*****************   Input Gate Weights and Biases   *****************//
        float LSTM_unit_weights::get_input_scaled_potential_memory_wxf()
        {
                return LSTM_weights[1].get_wx();
        }
        int LSTM_unit_weights::set_input_scaled_potential_memory_wxf(float new_weight)
        {
                return LSTM_weights[1].set_wx(new_weight);
        }

        float LSTM_unit_weights::get_input_scaled_potential_memory_whf()
        {
                return LSTM_weights[1].get_wh();
        }
        int LSTM_unit_weights::set_input_scaled_potential_memory_whf(float new_weight)
        {
                return LSTM_weights[1].set_wh(new_weight);
        }

        float LSTM_unit_weights::get_input_scaled_potential_memory_bf()
        {
                return LSTM_weights[1].get_b();
        }
        int LSTM_unit_weights::set_input_scaled_potential_memory_bf(float new_bias)
        {
                return LSTM_weights[1].set_b(new_bias);
        }


        float LSTM_unit_weights::get_input_potential_memory_wxf()
        {
                return LSTM_weights[2].get_wx();
        }
        int LSTM_unit_weights::set_input_potential_memory_wxf(float new_weight)
        {
                return LSTM_weights[2].set_wx(new_weight);
        }

        float LSTM_unit_weights::get_input_potential_memory_whf()
        {
                return LSTM_weights[2].get_wh();
        }
        int LSTM_unit_weights::set_input_potential_memory_whf(float new_weight)
        {
                return LSTM_weights[2].set_wh(new_weight);
        }

        float LSTM_unit_weights::get_input_potential_memory_bf()
        {
                return LSTM_weights[2].get_b();
        }
        int LSTM_unit_weights::set_input_potential_memory_bf(float new_bias)
        {
                return LSTM_weights[2].set_b(new_bias);
        }


//******************  Output Gate Weights and Biases  ******************//
        float LSTM_unit_weights::get_output_scaled_potential_memory_wxf()
        {
                return LSTM_weights[3].get_wx();
        }
        int LSTM_unit_weights::set_output_scaled_potential_memory_wxf(float new_weight)
        {
                return LSTM_weights[3].set_wx(new_weight);
        }

        float LSTM_unit_weights::get_output_scaled_potential_memory_whf()
        {
                return LSTM_weights[3].get_wh();
        }
        int LSTM_unit_weights::set_output_scaled_potential_memory_whf(float new_weight)
        {
                return LSTM_weights[3].set_wh(new_weight);
        }

        float LSTM_unit_weights::get_output_scaled_potential_memory_bf()
        {
                return LSTM_weights[3].get_b(); 
        }
        int LSTM_unit_weights::set_output_scaled_potential_memory_bf(float new_bias)
        {
                return LSTM_weights[3].set_b(new_bias);
        }