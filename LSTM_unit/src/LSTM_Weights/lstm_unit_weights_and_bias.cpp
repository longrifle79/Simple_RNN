#include "lstm_unit_weights_and_bias.h"



        Lstm_Unit_Weights::Lstm_Unit_Weights()
        {
                // Initialize Forget Gate
                LSTM_weights.push_back(Section_Weight_And_Bias());


                // Initialize Input Gate
                // Initialize Input Scaled Potential Memory
                LSTM_weights.push_back(Section_Weight_And_Bias()); 
                // Initialize Input Potential Memory
                LSTM_weights.push_back(Section_Weight_And_Bias()); 


                // Initialize Output Gate
                LSTM_weights.push_back(Section_Weight_And_Bias()); 
                // Initialize Output Scaled Potential Memory
                LSTM_weights.push_back(Section_Weight_And_Bias());
        }


//*****************   Forget Gate Weights and Biases   *****************//
        float Lstm_Unit_Weights::get_forget_gate_wxf()
        {
                return LSTM_weights[0].get_wx();
        }
        int Lstm_Unit_Weights::set_forget_gate_wxf(float new_weight)
        {
                return LSTM_weights[0].set_wx(new_weight);
        }

        float Lstm_Unit_Weights::get_forget_gate_whf()
        {
                return LSTM_weights[0].get_wh();
        }
        int Lstm_Unit_Weights::set_forget_gate_whf(float new_weight)
        {
                return LSTM_weights[0].set_wh(new_weight);
        }

        float Lstm_Unit_Weights::get_forget_gate_bf()
        {
                return LSTM_weights[0].get_b();
        }
        int Lstm_Unit_Weights::set_forget_gate_bf(float new_bias)
        {
                return LSTM_weights[0].set_b(new_bias);
        }


//*****************   Input Gate Weights and Biases   *****************//
        float Lstm_Unit_Weights::get_input_scaled_potential_memory_wxf()
        {
                return LSTM_weights[1].get_wx();
        }
        int Lstm_Unit_Weights::set_input_scaled_potential_memory_wxf(float new_weight)
        {
                return LSTM_weights[1].set_wx(new_weight);
        }

        float Lstm_Unit_Weights::get_input_scaled_potential_memory_whf()
        {
                return LSTM_weights[1].get_wh();
        }
        int Lstm_Unit_Weights::set_input_scaled_potential_memory_whf(float new_weight)
        {
                return LSTM_weights[1].set_wh(new_weight);
        }

        float Lstm_Unit_Weights::get_input_scaled_potential_memory_bf()
        {
                return LSTM_weights[1].get_b();
        }
        int Lstm_Unit_Weights::set_input_scaled_potential_memory_bf(float new_bias)
        {
                return LSTM_weights[1].set_b(new_bias);
        }


        float Lstm_Unit_Weights::get_input_potential_memory_wxf()
        {
                return LSTM_weights[2].get_wx();
        }
        int Lstm_Unit_Weights::set_input_potential_memory_wxf(float new_weight)
        {
                return LSTM_weights[2].set_wx(new_weight);
        }

        float Lstm_Unit_Weights::get_input_potential_memory_whf()
        {
                return LSTM_weights[2].get_wh();
        }
        int Lstm_Unit_Weights::set_input_potential_memory_whf(float new_weight)
        {
                return LSTM_weights[2].set_wh(new_weight);
        }

        float Lstm_Unit_Weights::get_input_potential_memory_bf()
        {
                return LSTM_weights[2].get_b();
        }
        int Lstm_Unit_Weights::set_input_potential_memory_bf(float new_bias)
        {
                return LSTM_weights[2].set_b(new_bias);
        }


//******************  Output Gate Weights and Biases  ******************//
        float Lstm_Unit_Weights::get_output_scaled_potential_memory_wxf()
        {
                return LSTM_weights[3].get_wx();
        }
        int Lstm_Unit_Weights::set_output_scaled_potential_memory_wxf(float new_weight)
        {
                return LSTM_weights[3].set_wx(new_weight);
        }

        float Lstm_Unit_Weights::get_output_scaled_potential_memory_whf()
        {
                return LSTM_weights[3].get_wh();
        }
        int Lstm_Unit_Weights::set_output_scaled_potential_memory_whf(float new_weight)
        {
                return LSTM_weights[3].set_wh(new_weight);
        }

        float Lstm_Unit_Weights::get_output_scaled_potential_memory_bf()
        {
                return LSTM_weights[3].get_b(); 
        }
        int Lstm_Unit_Weights::set_output_scaled_potential_memory_bf(float new_bias)
        {
                return LSTM_weights[3].set_b(new_bias);
        }