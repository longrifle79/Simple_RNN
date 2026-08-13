/*
 * multi_head_attention.cpp — implementation of causal multi-head self-attention
 *
 * Read compute_forward() slowly. It is only about forty lines of real work, and
 * those forty lines are the entire idea behind modern language models.
 */

#include "multi_head_attention.h"

#include <cmath>
#include <cassert>


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE
// ────────────────────────────────────────────────────────────────────────────────
void Multi_Head_Attention::initialize(int emb_dimension,
                                      int heads,
                                      float init_standard_deviation)
{
    embedding_dimension = emb_dimension;
    number_of_heads     = heads;

    // The channels are divided evenly among the heads. Model_Config::validate()
    // guarantees this divides cleanly, but assert it here too — a silent
    // truncation would produce a subtly broken model that still runs.
    assert(embedding_dimension % number_of_heads == 0);

    head_dimension = embedding_dimension / number_of_heads;

    // All four projections are C -> C.
    queryProjection.initialize(embedding_dimension, embedding_dimension, init_standard_deviation);
    keyProjection.initialize(embedding_dimension, embedding_dimension, init_standard_deviation);
    valueProjection.initialize(embedding_dimension, embedding_dimension, init_standard_deviation);
    outputProjection.initialize(embedding_dimension, embedding_dimension, init_standard_deviation);

    // One attention-weight matrix per head, sized later in compute_forward.
    cached_attention_weights.resize(number_of_heads);
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD PASS
//
// input:  [T, C]
// output: [T, C]
// ────────────────────────────────────────────────────────────────────────────────
Matrix Multi_Head_Attention::compute_forward(const Matrix& input)
{
    int number_of_tokens = input.rows;      // T

    // ── STEP 1: project the input into queries, keys and values ───────────────
    // Three different learned views of the same tokens. All are [T, C].
    Matrix all_queries = queryProjection.compute_forward(input);
    Matrix all_keys    = keyProjection.compute_forward(input);
    Matrix all_values  = valueProjection.compute_forward(input);

    // Where each head will write its own [T, head_dimension] result. Once all
    // heads are done this holds the concatenation of all of them.
    Matrix concatenated_heads(number_of_tokens, embedding_dimension);

    // The scaling factor from step 2 of the header explanation. Computed once
    // rather than inside the loop.
    float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));

    // ── STEP 2: run the attention procedure independently inside each head ────
    for (int h = 0; h < number_of_heads; ++h)
    {
        // Head h owns columns [h*head_dimension, (h+1)*head_dimension).
        int column_offset = h * head_dimension;

        // Pull out this head's slice of the queries, keys and values.
        // Each is [T, head_dimension].
        Matrix Q = matrix_slice_columns(all_queries, column_offset, head_dimension);
        Matrix K = matrix_slice_columns(all_keys,    column_offset, head_dimension);
        Matrix V = matrix_slice_columns(all_values,  column_offset, head_dimension);

        // ── STEP 2a: SCORE — every query against every key ────────────────────
        //
        //     scores = Q * K^T          [T, hd] * [hd, T]  ->  [T, T]
        //
        // scores(t, s) is the dot product of token t's query with token s's key:
        // "how relevant is token s to what token t is looking for".
        Matrix scores = matrix_multiply(Q, matrix_transpose(K));

        // ── STEP 2b: SCALE ────────────────────────────────────────────────────
        // Divide by sqrt(head_dimension) so softmax receives sensibly sized
        // numbers regardless of how wide the head is. See the header for why
        // this one line matters so much.
        scores = matrix_scale(scores, attention_scale);

        // ── STEP 2c: CAUSAL MASK + SOFTMAX ────────────────────────────────────
        //
        // Row t may only consider columns 0..t. Columns t+1 and beyond are
        // future tokens, which the model is forbidden to see.
        //
        // The usual implementation writes -infinity into the future positions so
        // exp() sends them to zero. We do something equivalent and simpler:
        // softmax over just the first (t+1) entries, and write plain 0 into the
        // rest. Same result, no magic numbers, and it is obvious from the code
        // that the future genuinely cannot contribute.
        for (int t = 0; t < number_of_tokens; ++t)
        {
            int number_of_visible_positions = t + 1;   // 0 through t inclusive

            softmax_row_prefix_in_place(scores, t, number_of_visible_positions);
        }

        // `scores` is now the attention weight matrix: every row sums to 1.
        // Stash it — this is the most interpretable object in the model.
        cached_attention_weights[h] = scores;

        // ── STEP 2d: MIX — weighted sum of the value vectors ──────────────────
        //
        //     head_output = attention_weights * V     [T, T] * [T, hd] -> [T, hd]
        //
        // Row t of the result is the sum of every value vector, each weighted by
        // how much attention token t decided to pay it. Because the masked
        // entries are exactly 0, future values contribute nothing.
        Matrix head_output = matrix_multiply(scores, V);

        // Drop this head's result into its slice of the combined matrix.
        matrix_write_columns(concatenated_heads, head_output, column_offset);
    }

    // ── STEP 3: mix the heads back together ───────────────────────────────────
    //
    // Up to this point the heads have been completely independent — head 0 knows
    // nothing about what head 3 found. This final C -> C projection lets the
    // network combine their findings into a single result.
    Matrix output = outputProjection.compute_forward(concatenated_heads);

    return output;
}


// ────────────────────────────────────────────────────────────────────────────────
// HOUSEKEEPING
// ────────────────────────────────────────────────────────────────────────────────
void Multi_Head_Attention::zero_gradients()
{
    queryProjection.zero_gradients();
    keyProjection.zero_gradients();
    valueProjection.zero_gradients();
    outputProjection.zero_gradients();
}


int Multi_Head_Attention::count_parameters() const
{
    return queryProjection.count_parameters()
         + keyProjection.count_parameters()
         + valueProjection.count_parameters()
         + outputProjection.count_parameters();
}
