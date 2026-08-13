/*
 * gpt_model.cpp — implementation of the full model
 */

#include "gpt_model.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE — build every component
// ────────────────────────────────────────────────────────────────────────────────
void Gpt_Model::initialize(const Model_Config& configuration)
{
    config = configuration;

    float init_std = config.initialization_standard_deviation;

    // ── Token embedding: one row per vocabulary symbol ────────────────────────
    tokenEmbedding.initialize(config.vocabulary_size,
                              config.embedding_dimension,
                              init_std);

    // ── Position embedding: one row per slot in the context window ────────────
    // Note it is sized by block_size, not by vocabulary size. The model has one
    // learned vector for "I am the 5th token", regardless of what that token is.
    positionEmbedding.initialize(config.block_size,
                                 config.embedding_dimension,
                                 init_std);

    // ── The stack of identical blocks ─────────────────────────────────────────
    transformerBlocks.resize(config.number_of_layers);

    for (int i = 0; i < config.number_of_layers; ++i)
    {
        transformerBlocks[i].initialize(config.embedding_dimension,
                                        config.number_of_heads,
                                        init_std);
    }

    // ── Final normalisation ───────────────────────────────────────────────────
    finalLayerNorm.initialize(config.embedding_dimension);

    // ── Output head: C -> V, one score per possible next token ────────────────
    languageModelHead.initialize(config.embedding_dimension,
                                 config.vocabulary_size,
                                 init_std);
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD PASS
// ────────────────────────────────────────────────────────────────────────────────
Matrix Gpt_Model::compute_forward(const std::vector<int>& token_indices)
{
    int number_of_tokens = static_cast<int>(token_indices.size());

    // The model physically cannot handle more tokens than it has position
    // embeddings for. This is the hard context limit.
    assert(number_of_tokens <= config.block_size);

    // ── STEP 1: look up what each token IS ────────────────────────────────────
    Matrix token_vectors = tokenEmbedding.lookup_rows(token_indices);

    // ── STEP 2: look up WHERE each token sits ─────────────────────────────────
    // The indices here are simply 0, 1, 2, ... — the slot numbers themselves.
    std::vector<int> position_indices =
        Embedding_Table::make_position_indices(number_of_tokens);

    Matrix position_vectors = positionEmbedding.lookup_rows(position_indices);

    // ── STEP 3: add them together ─────────────────────────────────────────────
    //
    // Addition, not concatenation. It looks almost too casual to work, but the
    // network has C dimensions to play with and readily learns to keep identity
    // information and position information in separable parts of that space.
    // Concatenating would cost extra width for no measured benefit.
    //
    // This sum is the start of what people call the RESIDUAL STREAM: a running
    // [T, C] representation that every block reads from and adds back into.
    Matrix x = matrix_add(token_vectors, position_vectors);

    // ── STEP 4: run the stack ─────────────────────────────────────────────────
    //
    // Every block takes [T, C] and returns [T, C]. Because the shape never
    // changes, adding depth is genuinely just adding another loop iteration.
    for (int i = 0; i < config.number_of_layers; ++i)
    {
        x = transformerBlocks[i].compute_forward(x);
    }

    // ── STEP 5: final normalisation ───────────────────────────────────────────
    x = finalLayerNorm.compute_forward(x);

    // ── STEP 6: project to vocabulary scores ──────────────────────────────────
    //
    // [T, C] * [C, V] -> [T, V]
    // Row t now holds one score per possible next token, for the token that
    // should follow position t.
    Matrix logits = languageModelHead.compute_forward(x);

    return logits;
}


// ────────────────────────────────────────────────────────────────────────────────
// CROSS ENTROPY LOSS
//
// The naive way is: softmax the logits, then take -log of the correct entry.
// That is mathematically right and numerically bad — exp() can overflow, and
// taking log of a tiny probability loses precision exactly where it matters.
//
// Instead we use the LOG-SUM-EXP form, which is algebraically identical:
//
//     -log( exp(z_correct) / sum_j exp(z_j) )
//   = -( z_correct - log( sum_j exp(z_j) ) )
//   = log( sum_j exp(z_j) ) - z_correct
//
// and we compute the log-sum-exp with the max subtracted out, the same
// stability trick softmax uses:
//
//     log( sum_j exp(z_j) ) = max + log( sum_j exp(z_j - max) )
//
// No overflow is possible, and no small number is ever logged.
// ────────────────────────────────────────────────────────────────────────────────
float Gpt_Model::compute_cross_entropy_loss(const Matrix& logits,
                                            const std::vector<int>& target_indices)
{
    int number_of_tokens = logits.rows;

    assert(static_cast<int>(target_indices.size()) == number_of_tokens);

    float total_loss = 0.0f;

    // Each position is its own independent prediction problem.
    for (int t = 0; t < number_of_tokens; ++t)
    {
        // ── Find the largest logit in this row, for stability ─────────────────
        float max_logit = logits.at(t, 0);

        for (int v = 1; v < logits.cols; ++v)
        {
            if (logits.at(t, v) > max_logit)
            {
                max_logit = logits.at(t, v);
            }
        }

        // ── Sum of exp(logit - max) over the whole vocabulary ─────────────────
        float sum_of_exponentials = 0.0f;

        for (int v = 0; v < logits.cols; ++v)
        {
            sum_of_exponentials += std::exp(logits.at(t, v) - max_logit);
        }

        // log-sum-exp, reassembled
        float log_sum_exp = max_logit + std::log(sum_of_exponentials);

        // The score the model gave to the token that actually came next
        int correct_token = target_indices[t];

        float correct_logit = logits.at(t, correct_token);

        // Loss for this position
        total_loss += (log_sum_exp - correct_logit);
    }

    // Average across positions so the number does not depend on sequence length
    return total_loss / number_of_tokens;
}


// ────────────────────────────────────────────────────────────────────────────────
// BATCH LOSS — average the loss over every sequence in the batch
// ────────────────────────────────────────────────────────────────────────────────
float Gpt_Model::compute_batch_loss(const std::vector<std::vector<int>>& input_batch,
                                    const std::vector<std::vector<int>>& target_batch)
{
    float total_loss = 0.0f;

    int batch_size = static_cast<int>(input_batch.size());

    // One sequence at a time — see the note at the top of gpt_model.h
    for (int b = 0; b < batch_size; ++b)
    {
        Matrix logits = compute_forward(input_batch[b]);

        total_loss += compute_cross_entropy_loss(logits, target_batch[b]);
    }

    return total_loss / batch_size;
}


// ────────────────────────────────────────────────────────────────────────────────
// HOUSEKEEPING
// ────────────────────────────────────────────────────────────────────────────────
void Gpt_Model::zero_gradients()
{
    tokenEmbedding.zero_gradients();
    positionEmbedding.zero_gradients();

    for (size_t i = 0; i < transformerBlocks.size(); ++i)
    {
        transformerBlocks[i].zero_gradients();
    }

    finalLayerNorm.zero_gradients();
    languageModelHead.zero_gradients();
}


int Gpt_Model::count_parameters() const
{
    int total = 0;

    total += tokenEmbedding.count_parameters();
    total += positionEmbedding.count_parameters();

    for (size_t i = 0; i < transformerBlocks.size(); ++i)
    {
        total += transformerBlocks[i].count_parameters();
    }

    total += finalLayerNorm.count_parameters();
    total += languageModelHead.count_parameters();

    return total;
}


// ────────────────────────────────────────────────────────────────────────────────
// PARAMETER BREAKDOWN
// Worth looking at: the feed-forward blocks dominate, not attention.
// ────────────────────────────────────────────────────────────────────────────────
void Gpt_Model::print_parameter_breakdown() const
{
    std::cout << "─── Parameter breakdown ───\n";

    std::cout << "  token embedding    : "
              << tokenEmbedding.count_parameters() << "\n";

    std::cout << "  position embedding : "
              << positionEmbedding.count_parameters() << "\n";

    // Every block is identical, so report one and multiply
    if (!transformerBlocks.empty())
    {
        const Transformer_Block& block = transformerBlocks[0];

        std::cout << "  per block          : " << block.count_parameters()
                  << "   (attention "
                  << block.multiHeadAttention.count_parameters()
                  << ", feed-forward "
                  << block.feedForward.count_parameters() << ")\n";

        std::cout << "  all " << transformerBlocks.size() << " blocks       : "
                  << block.count_parameters() * static_cast<int>(transformerBlocks.size())
                  << "\n";
    }

    std::cout << "  final layer norm   : "
              << finalLayerNorm.count_parameters() << "\n";

    std::cout << "  output head        : "
              << languageModelHead.count_parameters() << "\n";

    std::cout << "  TOTAL              : " << count_parameters() << "\n";
}
