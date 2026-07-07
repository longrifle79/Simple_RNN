#include <iostream>
#include "lstm_unit.h"

int main()
{
    // Values from your LSTM forward pass document (time step t-1)
    float x = 0.5f;                    // Current input
    float h_prev = 0.0f;               // Previous short-term memory (hidden state)
    float c_prev = 0.0f;               // Previous long-term memory (cell state)

    // Create LSTM object
    Lstm lstm;

    // Set weights using the values from your document
    // Forget Gate
    lstm.lstmUnitWeights.set_forget_gate_wxf(0.5f);
    lstm.lstmUnitWeights.set_forget_gate_whf(0.4f);
    lstm.lstmUnitWeights.set_forget_gate_bf(0.1f);

    // Input Gate - Scaled Potential Memory
    lstm.lstmUnitWeights.set_input_scaled_potential_memory_wxf(0.3f);
    lstm.lstmUnitWeights.set_input_scaled_potential_memory_whf(0.6f);
    lstm.lstmUnitWeights.set_input_scaled_potential_memory_bf(-0.2f);

    // Input Gate - Potential Memory
    lstm.lstmUnitWeights.set_input_potential_memory_wxf(0.7f);
    lstm.lstmUnitWeights.set_input_potential_memory_whf(0.2f);
    lstm.lstmUnitWeights.set_input_potential_memory_bf(0.3f);

    // Output Gate
    lstm.lstmUnitWeights.set_output_scaled_potential_memory_wxf(0.4f);
    lstm.lstmUnitWeights.set_output_scaled_potential_memory_whf(0.5f);
    lstm.lstmUnitWeights.set_output_scaled_potential_memory_bf(-0.1f);

    // Variables to store results
    float new_short_term_memory = 0.0f;
    float new_long_term_memory  = 0.0f;

    // Run the LSTM unit
    int result = lstm.compute_lstm_unit(
        x,
        h_prev,
        c_prev,
        new_short_term_memory,
        new_long_term_memory
    );

    // Print results
    std::cout << "=== LSTM Unit Test (t-1) ===\n";
    std::cout << "Input (x):                    " << x << "\n";
    std::cout << "Previous Short-Term Memory:   " << h_prev << "\n";
    std::cout << "Previous Long-Term Memory:    " << c_prev << "\n\n";

    std::cout << "New Long-Term Memory:         " << new_long_term_memory << "\n";
    std::cout << "New Short-Term Memory:        " << new_short_term_memory << "\n";
    std::cout << "Return code:                  " << result << "\n\n";

    // Expected values from your document
    std::cout << "=== Expected Values from Document ===\n";
    std::cout << "Expected New Long-Term Memory:  0.2787\n";
    std::cout << "Expected New Short-Term Memory: 0.1426\n";

    return 0;
}