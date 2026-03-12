#ifndef __RNN_HPP__
#define __RNN_HPP__

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

class SimpleRNN {
public:
    int input_size;   // usually 1 for univariate time series
    int hidden_size;  // number of hidden units (e.g., 32, 64)

    // Weights
    MatrixXd Wxh;     // input -> hidden
    MatrixXd Whh;     // hidden -> hidden (recurrent)
    MatrixXd Why;     // hidden -> output (for prediction)
    VectorXd bh;      // hidden bias
    VectorXd by;      // output bias

    SimpleRNN(int in_size, int hid_size);

    void initialize_weights(double stddev = 0.1);

    // Forward pass for one timestep
    VectorXd forward_step(const VectorXd& x_t, const VectorXd& h_prev) const;

    // Forward pass over entire sequence (returns hidden states + predictions)
    void forward_sequence(
        const vector<VectorXd>& inputs,
        vector<VectorXd>& hidden_states,
        vector<VectorXd>& outputs) const;

    // Simple loss (MSE) for one-step-ahead prediction
    double compute_loss(const vector<VectorXd>& targets, const vector<VectorXd>& predictions) const;

    // Placeholder for training (we'll add gradient descent / BPTT later)
    // void train(...);
};

#endif


