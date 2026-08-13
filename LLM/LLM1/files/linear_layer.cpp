/*
 * linear_layer.cpp — implementation of the fully connected layer
 */

#include "linear_layer.h"


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE
// ────────────────────────────────────────────────────────────────────────────────
void Linear_Layer::initialize(int in_dimension,
                              int out_dimension,
                              float init_standard_deviation)
{
    input_dimension  = in_dimension;
    output_dimension = out_dimension;

    // Allocate parameters at the right shape
    weights = Matrix(input_dimension, output_dimension);
    bias    = Matrix(1, output_dimension);

    // Weights get small random values so different output channels learn
    // different features. See Matrix::fill_with_random_normal for why.
    weights.fill_with_random_normal(0.0f, init_standard_deviation);

    // Biases start at exactly zero. A bias shifts every token identically, so
    // there is no symmetry for randomness to break, and zero means "no shift".
    bias.fill_with_zeros();

    // Gradient buffers mirror the parameter shapes, starting empty.
    gradient_weights = Matrix(input_dimension, output_dimension);
    gradient_bias    = Matrix(1, output_dimension);
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD
//
//     output = input * weights + bias
//
//   [T, in] * [in, out]  ->  [T, out],  then the [1, out] bias is broadcast
//   across all T rows.
//
// Read it row by row: each token's vector is dotted against each column of W to
// produce one output number. Every token goes through the same weights — the
// layer has no idea which position it is looking at, and does not care.
// ────────────────────────────────────────────────────────────────────────────────
Matrix Linear_Layer::compute_forward(const Matrix& input)
{
    // Remember the input for the backward pass (see header for why).
    cached_input = input;

    // The matrix multiply itself
    Matrix output = matrix_multiply(input, weights);

    // Add the bias to every row
    output = matrix_add_row_vector(output, bias);

    return output;
}


// ────────────────────────────────────────────────────────────────────────────────
// ZERO GRADIENTS
// Gradients accumulate across the sequences in a batch, so they must be cleared
// before each new batch or you would keep adding to last step's numbers.
// ────────────────────────────────────────────────────────────────────────────────
void Linear_Layer::zero_gradients()
{
    gradient_weights.fill_with_zeros();
    gradient_bias.fill_with_zeros();
}


// ────────────────────────────────────────────────────────────────────────────────
// COUNT PARAMETERS
// ────────────────────────────────────────────────────────────────────────────────
int Linear_Layer::count_parameters() const
{
    // Every weight plus every bias
    return (input_dimension * output_dimension) + output_dimension;
}
