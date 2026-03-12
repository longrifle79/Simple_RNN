#include <iostream>
#include <vector>
#include "rnn.h"

int main() {
    // Create RNN: input=1 (univariate), hidden=8 (small for testing)
    SimpleRNN rnn(1, 8);
    rnn.initialize_weights(0.08);  // small stddev to avoid exploding values

    // Dummy sequence: increasing numbers (like power trend)
    std::vector<VectorXd> inputs;
    for (int i = 1; i <= 10; ++i) {
        VectorXd x(1);
        x(0) = static_cast<double>(i);
        inputs.push_back(x);
    }

    std::vector<VectorXd> hidden_states, predictions;
    rnn.forward_sequence(inputs, hidden_states, predictions);

    std::cout << "Predictions (one-step ahead, naive):\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "t=" << i+1 << ": predicted = " << predictions[i](0)
                  << " (input was " << inputs[i](0) << ")\n";
    }

    // Dummy targets = next value (shifted)
    std::vector<VectorXd> targets;
    for (size_t i = 1; i < inputs.size(); ++i) {
        targets.push_back(inputs[i]);  // predict next
    }
    // Last prediction has no target → ignore or pad

    // Loss (will be high since untrained)
    double loss = rnn.compute_loss(targets, {predictions.begin(), predictions.end() - 1});
    std::cout << "\nMSE Loss (untrained): " << loss << std::endl;

    return 0;
}