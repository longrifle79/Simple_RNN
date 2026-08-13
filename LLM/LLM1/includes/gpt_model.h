/*
 * gpt_model.h — the complete decoder-only Transformer language model
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THE WHOLE ARCHITECTURE, TOP TO BOTTOM
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   token ids            [T]              e.g. [18, 47, 56, 57, 58]
 *        |
 *        |  token embedding lookup        "what symbol is this?"
 *        |  + position embedding          "where in the sequence am I?"
 *        v
 *   x                    [T, C]
 *        |
 *        |  Transformer block 1
 *        |  Transformer block 2
 *        |  ...            (number_of_layers of them)
 *        v
 *   x                    [T, C]
 *        |
 *        |  final layer norm
 *        |  language model head  (a Linear layer, C -> V)
 *        v
 *   logits               [T, V]
 *
 * Row t of the logits is the model's raw opinion about which token comes after
 * position t. Softmax turns it into probabilities.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * "DECODER-ONLY" — what that name actually means
 * ─────────────────────────────────────────────────────────────────────────────
 * The 2017 paper described an encoder-decoder for translation. The encoder read
 * the whole source sentence with no masking; the decoder generated output while
 * masked against the future, and also cross-attended to the encoder.
 *
 * GPT keeps only the decoder half and drops the cross-attention. What is left is
 * a causally-masked stack that reads its own output. That is the entire
 * architectural difference, and it is why our attention has the mask in it.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * ONE SEQUENCE AT A TIME — an important simplification
 * ─────────────────────────────────────────────────────────────────────────────
 * Real implementations push a whole batch through as a 3D tensor [B, T, C].
 * We process ONE sequence at a time, [T, C], and loop over the batch outside.
 * The result is identical; batching is purely a hardware efficiency trick.
 * Avoiding 3D tensors keeps every shape in the codebase two-dimensional and
 * every loop readable, which is worth far more to us than the speed.
 */

#ifndef __GPT_MODEL_H__
#define __GPT_MODEL_H__

#include "matrix.h"
#include "model_config.h"
#include "embedding_table.h"
#include "transformer_block.h"
#include "layer_norm.h"
#include "linear_layer.h"
#include "activation_functions.h"

#include <vector>
#include <string>


// ────────────────────────────────────────────────────────────────────────────────
// GPT_MODEL
// ────────────────────────────────────────────────────────────────────────────────
class Gpt_Model : public Activation_Functions
{
public:
    Model_Config config;

    // ────────────────────────────────
    // THE COMPONENTS, IN THE ORDER DATA FLOWS THROUGH THEM
    // ────────────────────────────────

    // "What symbol is this?" — one row per vocabulary entry
    Embedding_Table tokenEmbedding;

    // "Where am I in the sequence?" — one row per context slot
    Embedding_Table positionEmbedding;

    // The stack. A plain vector of blocks, all identically shaped.
    std::vector<Transformer_Block> transformerBlocks;

    // One last normalisation before reading out predictions. Without it the
    // final block's raw residual stream feeds straight into the output layer at
    // whatever scale it happens to be, which trains poorly.
    Layer_Norm finalLayerNorm;

    // The "language model head": projects each token's C-dimensional vector into
    // V scores, one per possible next token.
    Linear_Layer languageModelHead;

    Gpt_Model() = default;

    // Build every component from the config. Call once.
    void initialize(const Model_Config& configuration);

    // ────────────────────────────────
    // FORWARD PASS
    // token_indices is T ids (T must not exceed config.block_size).
    // Returns logits, shape [T, vocabulary_size].
    //
    // "Logits" are raw, unnormalised scores. They can be any real number,
    // positive or negative. Only after softmax do they become probabilities.
    // We keep them raw because the loss function can handle them far more
    // accurately in that form (see compute_cross_entropy_loss).
    // ────────────────────────────────
    Matrix compute_forward(const std::vector<int>& token_indices);

    // ────────────────────────────────
    // CROSS ENTROPY LOSS
    //
    // For each position t, the loss is -log(probability the model assigned to
    // the correct next token). Perfect confidence in the right answer gives
    // -log(1) = 0. Confidence in the wrong answer gives a large positive number.
    //
    // We return the average over all T positions.
    //
    // SANITY CHECK YOU SHOULD ALWAYS RUN: an untrained model has no idea what
    // comes next, so it spreads probability evenly across all V tokens. That
    // gives -log(1/V) = log(V). For a 65-symbol vocabulary that is about 4.17.
    // If your very first loss is not close to log(V), something is wrong before
    // you have even started training.
    // ────────────────────────────────
    float compute_cross_entropy_loss(const Matrix& logits,
                                     const std::vector<int>& target_indices);

    // Convenience: run a whole batch and return the mean loss over it.
    float compute_batch_loss(const std::vector<std::vector<int>>& input_batch,
                             const std::vector<std::vector<int>>& target_batch);

    void zero_gradients();

    int count_parameters() const;

    // Print a per-component parameter breakdown, so you can see where the
    // model's capacity actually lives.
    void print_parameter_breakdown() const;
};


#endif // __GPT_MODEL_H__
