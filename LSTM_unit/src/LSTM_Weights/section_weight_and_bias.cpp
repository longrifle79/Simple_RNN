#include "section_weight_and_bias.h"




        std::vector<float> block_weights;



        Section_Weight_And_Bias::Section_Weight_And_Bias()
        {
            // Initialize weights and bias to random values
            block_weights.push_back(getFloat(-1.0f, 1.0f)); // wh
            block_weights.push_back(getFloat(-1.0f, 1.0f)); // wx
            block_weights.push_back(getFloat(-1.0f, 1.0f)); // b
        }

        float Section_Weight_And_Bias::get_wh()
        {
            return block_weights[0];
        }

        int Section_Weight_And_Bias::set_wh(float new_weight)
        {
            block_weights[0] = new_weight;
            return 0;
        }

        float Section_Weight_And_Bias::get_wx()
        {
            return block_weights[1];
        }

        int Section_Weight_And_Bias::set_wx(float new_weight)
        {
            block_weights[1] = new_weight;
            return 0;
        }

        float Section_Weight_And_Bias::get_b()
        {
            return block_weights[2];
        }

        int Section_Weight_And_Bias::set_b(float new_bias)
        {
            block_weights[2] = new_bias;
            return 0;
        }








