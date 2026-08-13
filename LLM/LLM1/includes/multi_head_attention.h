/*
 * multi_head_attention.h — the mechanism that makes a Transformer a Transformer
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THE PROBLEM ATTENTION SOLVES
 * ─────────────────────────────────────────────────────────────────────────────
 * In our LSTM, information from step 1 reached step 50 by being squeezed through
 * 49 successive hidden states. Anything not preserved along the way was lost,
 * and the path was long and lossy. That is the vanishing gradient problem that
 * the gates were invented to fight.
 *
 * Attention throws that out. Every position can read directly from every earlier
 * position in one step. The path length between any two tokens is 1.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * QUERY, KEY, VALUE — the three roles every token plays
 * ─────────────────────────────────────────────────────────────────────────────
 * Each token's vector is projected three separate ways, by three separate
 * learned weight matrices:
 *
 *     QUERY  — "here is what I am looking for"
 *     KEY    — "here is what I contain, if anyone is looking"
 *     VALUE  — "here is what I will hand over if you pick me"
 *
 * The famous library analogy: the query is your search request, the keys are the
 * spine labels on the shelf, and the values are the books themselves. You match
 * your request against the labels to decide which books to take.
 *
 * The mechanism in four steps, for each position t:
 *
 *   1. SCORE   — dot every query against every key.
 *                score(t, s) = Q(t) · K(s). A large dot product means those two
 *                vectors point the same way, which means "this is relevant".
 *
 *   2. SCALE   — divide by sqrt(head_dimension).
 *                Without this, the dot product of two random head_dim-long
 *                vectors grows with head_dim, softmax gets huge inputs, and it
 *                saturates into a near one-hot distribution with almost no
 *                gradient. Dividing by sqrt(head_dim) keeps the variance at
 *                about 1 no matter how wide the head is.
 *
 *   3. MASK + SOFTMAX
 *              — a language model must never see the future. When predicting
 *                position 5 it may look at 0 through 5 and nothing beyond, or it
 *                would simply read off the answer and learn nothing. So position
 *                t only softmaxes over scores 0..t. The result is a set of
 *                weights that sum to 1: "how much of my attention goes to each
 *                earlier token".
 *
 *   4. MIX     — output(t) = the weighted sum of the VALUE vectors, using those
 *                attention weights.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY "MULTI-HEAD"
 * ─────────────────────────────────────────────────────────────────────────────
 * One attention pattern per layer is limiting — a token often needs to track
 * several unrelated things at once. So we split the C channels into
 * number_of_heads groups and run the whole procedure independently within each
 * group, then concatenate the results.
 *
 * Crucially the channels are SPLIT, not duplicated: with C=128 and 4 heads, each
 * head works in 32 dimensions. Four heads therefore cost the same as one big
 * head, but give four independent attention patterns. That is a genuinely good
 * deal, and it is why every Transformer does it.
 */

#ifndef __MULTI_HEAD_ATTENTION_H__
#define __MULTI_HEAD_ATTENTION_H__

#include "matrix.h"
#include "linear_layer.h"
#include "activation_functions.h"

#include <vector>


// ────────────────────────────────────────────────────────────────────────────────
// MULTI_HEAD_ATTENTION
//
// Inherits Activation_Functions so it can call softmax directly, exactly the way
// the LSTM gates inherited sigmoid and tanh.
// ────────────────────────────────────────────────────────────────────────────────
class Multi_Head_Attention : public Activation_Functions
{
public:
    int embedding_dimension;    // C
    int number_of_heads;        // how many independent attention patterns
    int head_dimension;         // C / number_of_heads

    // ────────────────────────────────
    // THE FOUR PROJECTIONS
    //
    // The first three turn each token vector into its query, key and value.
    // Note they are full C -> C layers, not C -> head_dim: one projection
    // produces the queries for ALL heads at once, and we slice it up afterwards.
    // That is both faster and closer to how real implementations do it.
    //
    // The fourth mixes the concatenated head outputs back together, so heads can
    // influence one another before the result leaves the layer.
    // ────────────────────────────────
    Linear_Layer queryProjection;
    Linear_Layer keyProjection;
    Linear_Layer valueProjection;
    Linear_Layer outputProjection;

    // ────────────────────────────────
    // CACHES FOR BACKPROPAGATION AND INSPECTION
    //
    // cached_attention_weights[h] is the [T, T] matrix of attention weights for
    // head h. Row t shows where token t looked. These are worth printing while
    // you learn — they are the most interpretable thing in the entire model, and
    // you can literally watch a head discover that quotation marks come in pairs.
    // ────────────────────────────────
    std::vector<Matrix> cached_attention_weights;

    Multi_Head_Attention() = default;

    void initialize(int emb_dimension, int heads, float init_standard_deviation);

    // Forward pass. Input and output are both [T, C].
    Matrix compute_forward(const Matrix& input);

    void zero_gradients();

    int count_parameters() const;
};


#endif // __MULTI_HEAD_ATTENTION_H__
