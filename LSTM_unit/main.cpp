#include <iostream>
#include "forget_gate.h"

int main()
{
    // Values from the LSTM forward pass document (time step t-1)
    float x_t_minus_1       = 0.5f;   // input
    float w_xf              = 0.5f;   // input weight for forget gate
    float h_t_minus_2       = 0.0f;   // previous short-term memory (hidden state)
    float w_hf              = 0.4f;   // hidden weight for forget gate
    float b_f               = 0.1f;   // forget gate bias

    ForgetGate forget_gate;

    // Call your function
    float forget_output = forget_gate.compute_forget_gate
    (
        x_t_minus_1,    // input
        w_xf,           // input_weight
        h_t_minus_2,    // prev_shrt_trm_mem
        w_hf,           // prev_shrt_trm_mem_weight
        b_f             // forget_bias
    );

    // Calculate the weighted sum manually so we can print it
    float weighted_sum = (x_t_minus_1 * w_xf) + 
                         (h_t_minus_2 * w_hf) + 
                         b_f;

    std::cout << "=== Forget Gate Test (t-1) ===\n";
    std::cout << "Input (x):                  " << x_t_minus_1 << "\n";
    std::cout << "Previous Hidden State (h):  " << h_t_minus_2 << "\n";
    std::cout << "Weighted Sum:               " << weighted_sum << "\n";
    std::cout << "Forget Gate Output (f):     " << forget_output << "\n";
    std::cout << "Expected (from document):   ~0.5866\n";

    return 0;
}