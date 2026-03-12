// ────────────────────────────────────────────────────────────────────────────────
// HEADER GUARD – prevents this file from being included multiple times
// If __RNN_HPP__ is already defined (meaning this file was included before),
// the preprocessor skips everything between #ifndef and #endif
// This avoids duplicate definition errors
// ────────────────────────────────────────────────────────────────────────────────
#ifndef __RNN_HPP__
#define __RNN_HPP__


// ────────────────────────────────────────────────────────────────────────────────
// INCLUDE NECESSARY HEADERS
// ────────────────────────────────────────────────────────────────────────────────

// Eigen library for efficient matrix and vector operations
// This is what lets us do fast linear algebra (matrix multiplication, etc.)
#include <Eigen/Dense>

// Standard library container for dynamic arrays
#include <vector>

// Random number generation (used in weight initialization)
#include <random>

// Input/output – used for error messages in some functions
#include <iostream>

// Math functions – we use std::tanh for activation
#include <cmath>


// ────────────────────────────────────────────────────────────────────────────────
// USING DECLARATIONS – make common Eigen and std types easier to write
// Instead of writing Eigen::MatrixXd everywhere, we can just write MatrixXd
// This is only safe inside this header because we control the namespace
// ────────────────────────────────────────────────────────────────────────────────
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


// ────────────────────────────────────────────────────────────────────────────────
// MAIN CLASS DEFINITION – SimpleRNN
// This is the core of our recurrent neural network
// It holds all the weights, biases, and functions needed to run the RNN
// ────────────────────────────────────────────────────────────────────────────────
class SimpleRNN
{
public:
    // ────────────────────────────────
    // MEMBER VARIABLES (the "memory" of the RNN object)
    // ────────────────────────────────

    // How many numbers come in at each timestep
    // Usually 1 for univariate time series (like power over time)
    int input_size;

    // Number of neurons/units in the hidden layer
    // Controls how much "memory" and capacity the RNN has
    // Larger = more powerful but slower and harder to train
    int hidden_size;

    // ────────────────────────────────
    // WEIGHTS – the learnable parameters of the network
    // These are matrices that transform data between layers
    // ────────────────────────────────

    // Input to hidden weights
    // Shape: hidden_size rows × input_size columns
    // Turns input vector into contribution to hidden state
    MatrixXd Wxh;

    // Recurrent (hidden to hidden) weights
    // Shape: hidden_size × hidden_size (square matrix)
    // This is what gives the RNN its "memory" – it connects previous hidden state to current
    MatrixXd Whh;

    // Hidden to output weights
    // Shape: 1 × hidden_size (because we predict one number)
    // Projects the hidden state into the final prediction space
    MatrixXd Why;

    // ────────────────────────────────
    // BIASES – small learnable offsets added after matrix multiplications
    // Help the network shift activations to better fit data
    // ────────────────────────────────

    // Bias for hidden layer – one value per hidden neuron
    VectorXd bh;

    // Bias for output – one value (since output size = 1)
    VectorXd by;

    // ────────────────────────────────
    // CONSTRUCTOR DECLARATION
    // Called when you write: SimpleRNN my_rnn(1, 32);
    // ────────────────────────────────
    SimpleRNN(int in_size, int hid_size);

    // ────────────────────────────────
    // Initializes all weights and biases with small random values
    // Default stddev = 0.1 (small to prevent exploding gradients)
    // ────────────────────────────────
    void initialize_weights(double stddev = 0.1);

    // ────────────────────────────────
    // FORWARD STEP – computes ONE timestep of the RNN
    // Takes current input x_t and previous hidden state h_prev
    // Returns new hidden state h_t
    // Marked const because it doesn't modify the RNN's weights
    // ────────────────────────────────
    VectorXd forward_step(const VectorXd& x_t, const VectorXd& h_prev) const;

    // ────────────────────────────────
    // FORWARD SEQUENCE – runs the RNN over an entire sequence of inputs
    // Fills hidden_states and outputs by reference (caller gets the results)
    // Also const – only reads, doesn't change weights
    // ────────────────────────────────
    void forward_sequence(
        const vector<VectorXd>& inputs,           // All input vectors x1, x2, ..., xT
        vector<VectorXd>& hidden_states,          // Will be filled with h1, h2, ..., hT
        vector<VectorXd>& outputs) const;         // Will be filled with predictions ŷ1, ŷ2, ..., ŷT

    // ────────────────────────────────
    // COMPUTE LOSS – calculates Mean Squared Error between predictions and targets
    // Used to measure how wrong the model is
    // Also const – pure calculation, no changes to object
    // ────────────────────────────────
    double compute_loss(const vector<VectorXd>& targets,
                        const vector<VectorXd>& predictions) const;

    // ────────────────────────────────
    // PLACEHOLDER FOR FUTURE TRAINING FUNCTION
    // We'll eventually add backpropagation through time (BPTT) here
    // For now it's commented out so we can focus on forward pass first
    // ────────────────────────────────
    // void train(...);  // To be implemented later
};


// ────────────────────────────────────────────────────────────────────────────────
// END OF HEADER GUARD – everything after this is ignored if file is included again
// ────────────────────────────────────────────────────────────────────────────────
#endif