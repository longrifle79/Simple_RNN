#ifndef __SECTION_WEIGHT_AND_BIAS_H__
#define __SECTION_WEIGHT_AND_BIAS_H__

#include "randomize_weight.h"


class section_weight_and_bias:RandomizeWeight
{
    private:
        std::vector<float> block_weights;

    public:

        section_weight_and_bias();

        float get_wh();
        int set_wh(float new_weight);

        float get_wx();
        int set_wx(float new_weight);

        float get_b();
        int set_b(float new_bias);
};

#endif // __SECTION_WEIGHT_AND_BIAS_H__

