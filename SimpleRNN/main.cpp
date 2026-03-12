// Include standard library for input/output (std::cout)
#include <iostream>

// Include vector container from standard library
// We use std::vector to hold sequences of data
#include <vector>

// Include our RNN class definition
// This tells the compiler what SimpleRNN is and what functions it has
#include "rnn.h"


// ────────────────────────────────────────────────────────────────────────────────
// MAIN FUNCTION – the entry point of the entire program
// When you run the executable, this is where execution starts
// ────────────────────────────────────────────────────────────────────────────────
int main()
{
    // Create an instance (object) of our SimpleRNN class
    // input_size = 1 because we're predicting one number at a time (univariate power data)
    // hidden_size = 8 → small number of hidden neurons just for testing
    // This calls the constructor we defined in rnn.cpp
    SimpleRNN rnn(1, 8);

    // Initialize all weights and biases with small random values
    // We use a small standard deviation (0.08) to prevent the network from starting
    // with very large numbers that could cause exploding gradients later
    rnn.initialize_weights(0.08);

    // ────────────────────────────────
    // Create a dummy input sequence for testing
    // This simulates a simple increasing trend (like steadily rising power usage)
    // ────────────────────────────────
    std::vector<VectorXd> inputs;

    // Loop from 1 to 10 to create 10 timesteps
    for (int i = 1; i <= 10; ++i)
    {
        // Create a VectorXd with size 1 (because input_size = 1)
        VectorXd x(1);

        // Put the current number (1, 2, 3, ..., 10) into the vector
        // static_cast<double>(i) converts integer i to double
        x(0) = static_cast<double>(i);

        // Add this input vector to our sequence
        inputs.push_back(x);
    }

    // Create two empty vectors that forward_sequence will fill
    // hidden_states will hold all hidden states h1, h2, ..., h10
    // predictions will hold all model outputs ŷ1, ŷ2, ..., ŷ10
    std::vector<VectorXd> hidden_states;
    std::vector<VectorXd> predictions;

    // Run the entire sequence through the RNN in one call
    // This calls forward_sequence() from rnn.cpp
    // It fills hidden_states and predictions by reference (modifies them directly)
    rnn.forward_sequence(inputs, hidden_states, predictions);

    // ────────────────────────────────
    // Print the predictions so we can see what the untrained model outputs
    // ────────────────────────────────
    std::cout << "Predictions (one-step ahead, naive):\n";

    // Loop through every prediction we just made
    // Using int instead of size_t (safe here because vector is small)
    for (int i = 0; i < static_cast<int>(predictions.size()); ++i)
    {
        // Print timestep number (starting from 1 for readability)
        std::cout << "t=" << i + 1;

        // Print what the model predicted (access element 0 because output size = 1)
        std::cout << ": predicted = " << predictions[i](0);

        // Also show what the actual input was at this timestep
        std::cout << " (input was " << inputs[i](0) << ")\n";
    }

    // ────────────────────────────────
    // Create target values for loss calculation
    // For one-step-ahead prediction, target at time t is the input at time t+1
    // ────────────────────────────────
    std::vector<VectorXd> targets;

    // Start from the second input (index 1) to the end
    // Using int instead of size_t
    for (int i = 1; i < static_cast<int>(inputs.size()); ++i)
    {
        // The target for timestep i-1 is the actual value at timestep i
        targets.push_back(inputs[i]);
    }
    // Note: We have one fewer target than predictions
    // That's why we ignore the last prediction when computing loss

    // ────────────────────────────────
    // Compute and print the loss (how wrong our predictions are)
    // Since the model is untrained, expect a large number
    // ────────────────────────────────
    // Create a temporary vector that excludes the very last prediction
    // (because we have no target for the final timestep)
    std::vector<VectorXd> predictions_for_loss(predictions.begin(), predictions.end() - 1);

    // Call the loss function from rnn.cpp
    double loss = rnn.compute_loss(targets, predictions_for_loss);

    // Print the result with a newline before it for readability
    std::cout << "\nMSE Loss (untrained): " << loss << std::endl;

    // Return 0 to tell the operating system "program finished successfully"
    // Non-zero would indicate an error
    return 0;
}