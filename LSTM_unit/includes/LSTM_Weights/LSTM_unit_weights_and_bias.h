#ifndef __LSTM_UNIT_WEIGHTS_AND_BIAS_H__
#define __LSTM_UNIT_WEIGHTS_AND_BIAS_H__

#include "section_weight_and_bias.h"



class LSTM_unit_weights:section_weight_and_bias
{
    private:
        std::vector<section_weight_and_bias> LSTM_weights;

    public:
        LSTM_unit_weights();


//*****************   Forget Gate Weights and Biases   *****************//
        float get_forget_gate_wxf();
        int set_forget_gate_wxf(float new_weight);

        float get_forget_gate_whf();
        int set_forget_gate_whf(float new_weight);

        float get_forget_gate_bf();
        int set_forget_gate_bf(float new_bias);


//*****************   Input Gate Weights and Biases   *****************//
        float get_input_scaled_potential_memory_wxf();
        int set_input_scaled_potential_memory_wxf(float new_weight);

        float get_input_scaled_potential_memory_whf();
        int set_input_scaled_potential_memory_whf(float new_weight);

        float get_input_scaled_potential_memory_bf();
        int set_input_scaled_potential_memory_bf(float new_bias);


        float get_input_potential_memory_wxf();
        int set_input_potential_memory_wxf(float new_weight);

        float get_input_potential_memory_whf();
        int set_input_potential_memory_whf(float new_weight);

        float get_input_potential_memory_bf();
        int set_input_potential_memory_bf(float new_bias);


//******************  Output Gate Weights and Biases  ******************//
        float get_output_scaled_potential_memory_wxf();
        int set_output_scaled_potential_memory_wxf(float new_weight);

        float get_output_scaled_potential_memory_whf();
        int set_output_scaled_potential_memory_whf(float new_weight);

        float get_output_scaled_potential_memory_bf();
        int set_output_scaled_potential_memory_bf(float new_bias);

};

#endif // __LSTM_UNIT_WEIGHTS_AND_BIAS_H__

