/*
 * feed_forward.cpp — implementation of the position-wise feed-forward network
 */

#include "feed_forward.h"


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE
// ────────────────────────────────────────────────────────────────────────────────
void Feed_Forward::initialize(int emb_dimension, float init_standard_deviation)
{
    embedding_dimension = emb_dimension;

    // The 4x expansion factor from the original paper
    hidden_dimension = 4 * embedding_dimension;

    expandLayer.initialize(embedding_dimension, hidden_dimension, init_standard_deviation);
    contractLayer.initialize(hidden_dimension, embedding_dimension, init_standard_deviation);
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD
//
//     hidden    = input * W1 + b1          [T, C]  -> [T, 4C]
//     activated = gelu(hidden)             elementwise
//     output    = activated * W2 + b2      [T, 4C] -> [T, C]
// ────────────────────────────────────────────────────────────────────────────────
Matrix Feed_Forward::compute_forward(const Matrix& input)
{
    // ── Step 1: expand to the wider hidden space ──────────────────────────────
    Matrix hidden = expandLayer.compute_forward(input);

    // Save the pre-activation values — gelu_derivative needs them later.
    cached_pre_activation = hidden;

    // ── Step 2: apply the non-linearity, element by element ───────────────────
    //
    // This is the only step in the entire block that is not a matrix multiply,
    // and it is the only reason the block can do anything a single linear layer
    // could not. Two stacked linear layers with nothing between them collapse
    // into one linear layer; GELU is what prevents that collapse.
    for (size_t i = 0; i < hidden.values.size(); ++i)
    {
        hidden.values[i] = gelu(hidden.values[i]);
    }

    // ── Step 3: contract back to the model width ──────────────────────────────
    Matrix output = contractLayer.compute_forward(hidden);

    return output;
}


// ────────────────────────────────────────────────────────────────────────────────
// HOUSEKEEPING
// ────────────────────────────────────────────────────────────────────────────────
void Feed_Forward::zero_gradients()
{
    expandLayer.zero_gradients();
    contractLayer.zero_gradients();
}


int Feed_Forward::count_parameters() const
{
    return expandLayer.count_parameters() + contractLayer.count_parameters();
}
