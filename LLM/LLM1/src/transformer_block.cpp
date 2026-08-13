/*
 * transformer_block.cpp — implementation of one Transformer layer
 */

#include "transformer_block.h"


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE
// ────────────────────────────────────────────────────────────────────────────────
void Transformer_Block::initialize(int embedding_dimension,
                                   int number_of_heads,
                                   float init_standard_deviation)
{
    normBeforeAttention.initialize(embedding_dimension);

    multiHeadAttention.initialize(embedding_dimension,
                                  number_of_heads,
                                  init_standard_deviation);

    normBeforeFeedForward.initialize(embedding_dimension);

    feedForward.initialize(embedding_dimension, init_standard_deviation);
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD
//
//     x = x + attention(layer_norm(x))
//     x = x + feed_forward(layer_norm(x))
//
// Four lines of real code. Everything the Transformer does, it does here.
// ────────────────────────────────────────────────────────────────────────────────
Matrix Transformer_Block::compute_forward(const Matrix& input)
{
    // ── First half: communicate ───────────────────────────────────────────────

    // Normalise a COPY of the input. The original is untouched, because it still
    // has to be added back in a moment.
    Matrix normalized_for_attention = normBeforeAttention.compute_forward(input);

    // Gather information from earlier positions
    Matrix attention_output = multiHeadAttention.compute_forward(normalized_for_attention);

    // The residual connection. Note we add the ORIGINAL input, not the
    // normalised one — that is what keeps the highway clean.
    Matrix after_attention = matrix_add(input, attention_output);

    // ── Second half: compute ──────────────────────────────────────────────────

    Matrix normalized_for_feed_forward = normBeforeFeedForward.compute_forward(after_attention);

    Matrix feed_forward_output = feedForward.compute_forward(normalized_for_feed_forward);

    // Second residual connection
    Matrix output = matrix_add(after_attention, feed_forward_output);

    return output;
}


// ────────────────────────────────────────────────────────────────────────────────
// HOUSEKEEPING — just pass the request down to the four components
// ────────────────────────────────────────────────────────────────────────────────
void Transformer_Block::zero_gradients()
{
    normBeforeAttention.zero_gradients();
    multiHeadAttention.zero_gradients();
    normBeforeFeedForward.zero_gradients();
    feedForward.zero_gradients();
}


int Transformer_Block::count_parameters() const
{
    return normBeforeAttention.count_parameters()
         + multiHeadAttention.count_parameters()
         + normBeforeFeedForward.count_parameters()
         + feedForward.count_parameters();
}
