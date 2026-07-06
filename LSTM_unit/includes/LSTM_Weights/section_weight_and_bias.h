#ifndef __SECTION_WEIGHT_AND_BIAS_H__
#define __SECTION_WEIGHT_AND_BIAS_H__

#include "randomize_weight.h"


class Section_Weight_And_Bias:Randomize_Weight
{
    private:
        std::vector<float> block_weights;

    public:

        Section_Weight_And_Bias();

        float get_wh();
        int set_wh(float new_weight);

        float get_wx();
        int set_wx(float new_weight);

        float get_b();
        int set_b(float new_bias);
};

#endif // __SECTION_WEIGHT_AND_BIAS_H__

