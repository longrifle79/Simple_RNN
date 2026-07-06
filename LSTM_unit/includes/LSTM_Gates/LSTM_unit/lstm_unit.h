#ifndef __LSTM_UNIT_H__
#define __LSTM_UNIT_H__


#include "activation_functions.h"
#include "lstm_unit_weights_and_bias.h"
#include "forget_gate.h"
#include "input_gate.h"
#include "output_gate.h"





class Lstm
{
    private:
        Forget_Gate forgetGate;
        Input_Gate inputGate;
        Output_Gate outputGate;
        Lstm_Unit_Weights lstmUnitWeights;

    public:
        Lstm() = default;

        int compute_lstm_unit(    float input, 
                                    float short_term_memory, 
                                    float long_term_memory, 
                                    Lstm_Unit_Weights lstmUnitWeights,
                                    float &new_short_term_memory,
                                    float &new_long_term_memory);



};




#endif // __LSTM_UNIT_H__

