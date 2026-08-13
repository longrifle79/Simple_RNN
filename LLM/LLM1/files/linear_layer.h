/*
 * linear_layer.h — a fully connected layer: output = input * W + b
 *
 * This is the workhorse of the whole Transformer. Attention uses four of them
 * (query, key, value, output projections), the feed-forward block uses two, and
 * the final vocabulary head is one more. Get this right and most of the model is
 * already written.
 *
 * SHAPES:
 *     input   [T, input_dimension]
 *     weights [input_dimension, output_dimension]
 *     bias    [1, output_dimension]        (broadcast across all T rows)
 *     output  [T, output_dimension]
 *
 * Note the weight layout: [in, out], so the multiply reads input * W. Many
 * textbooks write W * x with W shaped [out, in]. Both are correct; this ordering
 * keeps the token dimension first everywhere, which means every matrix in the
 * project has T as its row count. One convention, no mental gymnastics.
 */

#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

#include "matrix.h"


// ────────────────────────────────────────────────────────────────────────────────
// LINEAR_LAYER
// ────────────────────────────────────────────────────────────────────────────────
class Linear_Layer
{
public:
    // ────────────────────────────────
    // SHAPE
    // ────────────────────────────────
    int input_dimension;
    int output_dimension;

    // ────────────────────────────────
    // LEARNABLE PARAMETERS
    // These are the numbers that training actually changes.
    // ────────────────────────────────
    Matrix weights;     // [input_dimension, output_dimension]
    Matrix bias;        // [1, output_dimension]

    // ────────────────────────────────
    // GRADIENT BUFFERS
    //
    // Same shape as the parameters above. After a backward pass, gradient_weights
    // holds "how much the loss would change if I nudged this weight".
    //
    // They are declared now, in the forward-pass stage, because building the
    // backward pass later becomes a matter of filling in a function rather than
    // restructuring every class. Right now they sit at zero.
    // ────────────────────────────────
    Matrix gradient_weights;
    Matrix gradient_bias;

    // ────────────────────────────────
    // CACHED INPUT
    //
    // Backpropagation through a linear layer needs the input that produced the
    // output: the gradient with respect to W is input^T * gradient_of_output.
    // So the forward pass must remember what it saw. Storing activations is the
    // main reason training uses far more memory than inference.
    // ────────────────────────────────
    Matrix cached_input;

    Linear_Layer() = default;

    // Allocate and randomise. Called once, at model construction.
    // Biases start at zero — there is no symmetry to break in a bias, and zero
    // is the neutral starting point.
    void initialize(int in_dimension, int out_dimension, float init_standard_deviation);

    // Forward pass: output = input * weights + bias
    Matrix compute_forward(const Matrix& input);

    // Reset gradient buffers to zero. Called before each backward pass, because
    // gradients ACCUMULATE — we sum them over every sequence in the batch.
    void zero_gradients();

    // How many learnable numbers this layer owns. Used for the parameter count.
    int count_parameters() const;
};


#endif // __LINEAR_LAYER_H__
