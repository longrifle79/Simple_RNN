/*
 * feed_forward.h — the "thinking" half of a Transformer block
 *
 * Attention moves information BETWEEN tokens. This block does the processing
 * WITHIN each token. Every token goes through it separately and identically —
 * it has no idea what its neighbours are doing.
 *
 * The structure is deliberately simple: expand, apply a non-linearity, contract.
 *
 *     [T, C]  --expand-->  [T, 4C]  --GELU-->  [T, 4C]  --contract-->  [T, C]
 *
 * WHY EXPAND TO 4x AND COME BACK:
 * A non-linearity is only useful if there is room to work in. Widening to 4C
 * gives the layer a much larger space in which to separate patterns before
 * squeezing the useful part back down to C. The factor of 4 comes from the
 * original Transformer paper and has stuck ever since — it is a convention that
 * works well, not a law.
 *
 * Worth knowing: this block holds about two thirds of the parameters in a
 * typical Transformer. Attention gets the attention, but most of the model's
 * stored knowledge lives here.
 */

#ifndef __FEED_FORWARD_H__
#define __FEED_FORWARD_H__

#include "matrix.h"
#include "linear_layer.h"
#include "activation_functions.h"


// ────────────────────────────────────────────────────────────────────────────────
// FEED_FORWARD
// Inherits Activation_Functions so gelu() can be called as if it were a method,
// the same pattern the LSTM gates used for sigmoid().
// ────────────────────────────────────────────────────────────────────────────────
class Feed_Forward : public Activation_Functions
{
public:
    int embedding_dimension;    // C
    int hidden_dimension;       // 4 * C

    Linear_Layer expandLayer;     // C  -> 4C
    Linear_Layer contractLayer;   // 4C -> C

    // ────────────────────────────────
    // CACHE FOR BACKPROPAGATION
    // The derivative of GELU must be evaluated at the value that went INTO it,
    // not the value that came out, so we keep the pre-activation matrix.
    // ────────────────────────────────
    Matrix cached_pre_activation;   // [T, 4C]

    Feed_Forward() = default;

    void initialize(int emb_dimension, float init_standard_deviation);

    Matrix compute_forward(const Matrix& input);

    void zero_gradients();

    int count_parameters() const;
};


#endif // __FEED_FORWARD_H__
