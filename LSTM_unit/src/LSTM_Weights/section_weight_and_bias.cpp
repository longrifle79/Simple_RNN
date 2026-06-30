#include "section_weight_and_bias.h"




        std::vector<float> block_weights;



        section_weight_and_bias::section_weight_and_bias()
        {
            // Initialize weights and bias to random values
            block_weights.push_back(getFloat(-1.0f, 1.0f)); // wh
            block_weights.push_back(getFloat(-1.0f, 1.0f)); // wx
            block_weights.push_back(getFloat(-1.0f, 1.0f)); // b
        }

        float section_weight_and_bias::get_wh()
        {
            return block_weights[0];
        }

        int section_weight_and_bias::set_wh(float new_weight)
        {
            block_weights[0] = new_weight;
            return 0;
        }

        float section_weight_and_bias::get_wx()
        {
            return block_weights[1];
        }

        int section_weight_and_bias::set_wx(float new_weight)
        {
            block_weights[1] = new_weight;
            return 0;
        }

        float section_weight_and_bias::get_b()
        {
            return block_weights[2];
        }

        int section_weight_and_bias::set_b(float new_bias)
        {
            block_weights[2] = new_bias;
            return 0;
        }








