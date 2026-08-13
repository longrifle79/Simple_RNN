/*
 * transformer_block.h — one complete layer of the model
 *
 * A Transformer is just this block repeated N times. Nothing else changes with
 * depth. If you understand this file, you understand the architecture.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THE TWO HALVES
 * ─────────────────────────────────────────────────────────────────────────────
 *   1. Attention    — "gather relevant information from earlier tokens"
 *   2. Feed-forward — "think about what I now have"
 *
 * Communicate, then compute. Every block does both, in that order.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * RESIDUAL CONNECTIONS — output = x + sublayer(x)
 * ─────────────────────────────────────────────────────────────────────────────
 * Notice we ADD the sublayer's result to the input rather than replacing it.
 * Two reasons, both important:
 *
 *   - Each sublayer only has to learn a CORRECTION to what came before, not a
 *     whole new representation from scratch. Doing nothing is easy: output
 *     zeros. That makes deep stacks much easier to optimise.
 *
 *   - Gradients flow backwards through the addition completely unchanged. The
 *     "+ x" gives every layer a direct connection to the loss, no matter how
 *     deep it sits. This is the same trick ResNets use, and it is the reason a
 *     96-layer model is trainable at all.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PRE-NORM vs POST-NORM — a deliberate choice
 * ─────────────────────────────────────────────────────────────────────────────
 * We normalise BEFORE each sublayer and add the raw input back:
 *
 *     x = x + attention(layer_norm(x))
 *     x = x + feed_forward(layer_norm(x))
 *
 * The original 2017 paper did the opposite (normalise after the addition). That
 * version needs a learning-rate warmup schedule or it diverges. The pre-norm
 * arrangement, used by GPT-2 onward, trains stably without any of that — which
 * is exactly what we want while learning. The residual path stays a clean,
 * unmodified highway from input to output.
 */

#ifndef __TRANSFORMER_BLOCK_H__
#define __TRANSFORMER_BLOCK_H__

#include "matrix.h"
#include "layer_norm.h"
#include "multi_head_attention.h"
#include "feed_forward.h"


// ────────────────────────────────────────────────────────────────────────────────
// TRANSFORMER_BLOCK
// ────────────────────────────────────────────────────────────────────────────────
class Transformer_Block
{
public:
    // Composition, exactly like the Lstm class holding its three gates.
    Layer_Norm           normBeforeAttention;
    Multi_Head_Attention multiHeadAttention;
    Layer_Norm           normBeforeFeedForward;
    Feed_Forward         feedForward;

    Transformer_Block() = default;

    void initialize(int embedding_dimension,
                    int number_of_heads,
                    float init_standard_deviation);

    // Input and output are both [T, C] — a block never changes the shape.
    // That is precisely why blocks can be stacked to any depth.
    Matrix compute_forward(const Matrix& input);

    void zero_gradients();

    int count_parameters() const;
};


#endif // __TRANSFORMER_BLOCK_H__
