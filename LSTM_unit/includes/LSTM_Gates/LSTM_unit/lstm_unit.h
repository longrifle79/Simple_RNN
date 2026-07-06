#ifndef __LSTM_UNIT_H__
#define __LSTM_UNIT_H__


#include "activation_functions.h"
#include "LSTM_unit_weights_and_bias.h"
#include "forget_gate.h"
#include "input_gate.h"
#include "output_gate.h"





class Lstm
{
    private:
        ForgetGate forget_gate;
        InputGate input_gate;
        OutputGate output_gate;
        LSTM_unit_weights lstm_unit_weights;

    public:
        Lstm() = default;

        int compute_lstm_unit(    float input, 
                                    float short_term_memory, 
                                    float long_term_memory, 
                                    LSTMUnitWeights lstm_unit_weights,
                                    float &new_short_term_memory,
                                    float &new_long_term_memory);



};




#endif // __LSTM_UNIT_H__

