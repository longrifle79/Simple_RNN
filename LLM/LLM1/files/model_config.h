/*
 * model_config.h — every hyperparameter in one place
 *
 * WHY A CONFIG OBJECT INSTEAD OF #defines:
 * The LSTM project hard-coded its sizes, which was fine for one fixed unit. Here
 * you will want to try a bigger model, a longer context, more heads. Gathering
 * every knob into one struct means experiments are a one-line change, and every
 * layer can be handed the same config rather than five loose integers.
 *
 * This is the one place in the project where public members with no getters are
 * clearly the right call — it is a bag of settings, not an object with behaviour.
 */

#ifndef __MODEL_CONFIG_H__
#define __MODEL_CONFIG_H__

#include <iostream>


// ────────────────────────────────────────────────────────────────────────────────
// MODEL_CONFIG
// The defaults below produce a model of roughly 0.8 million parameters, which
// trains on a normal desktop CPU. Both NUCs will handle it comfortably.
// ────────────────────────────────────────────────────────────────────────────────
class Model_Config
{
public:
    // ────────────────────────────────
    // VOCABULARY SIZE (V)
    // How many distinct tokens exist. Set from the tokenizer, not by hand —
    // about 65 for Tiny Shakespeare at character level. It fixes the width of
    // both the embedding table and the final output layer.
    // ────────────────────────────────
    int vocabulary_size = 0;

    // ────────────────────────────────
    // BLOCK SIZE (T) — the context length
    // The most tokens the model can ever look back at. Position 40 can attend to
    // positions 0..40; it has no mechanism whatsoever to see position 41.
    //
    // Attention cost grows with T*T, so doubling the context roughly quadruples
    // the attention work. 64 characters is enough to learn word shapes, line
    // breaks and speaker names in Shakespeare.
    // ────────────────────────────────
    int block_size = 64;

    // ────────────────────────────────
    // EMBEDDING DIMENSION (C) — the width of the model
    // Every token is represented by a vector of this many numbers, all the way
    // through the network. Bigger C means more capacity per token.
    // MUST divide evenly by number_of_heads.
    // ────────────────────────────────
    int embedding_dimension = 128;

    // ────────────────────────────────
    // NUMBER OF ATTENTION HEADS
    // The C dimensions are split into this many independent groups. Each head
    // gets embedding_dimension / number_of_heads channels and learns its own
    // notion of what to pay attention to — one head might track quotation marks
    // while another tracks the previous vowel. Splitting instead of duplicating
    // means multi-head attention costs the same as single-head.
    // ────────────────────────────────
    int number_of_heads = 4;

    // ────────────────────────────────
    // NUMBER OF TRANSFORMER BLOCKS — the depth of the model
    // Each block is one round of "gather context, then think about it".
    // ────────────────────────────────
    int number_of_layers = 4;

    // ────────────────────────────────
    // TRAINING SETTINGS
    // ────────────────────────────────

    // How many independent sequences we process before updating the weights.
    // Bigger batches give a less noisy estimate of the gradient direction.
    int batch_size = 32;

    // Step size for the optimizer. 3e-4 is the standard starting point for Adam
    // on Transformers; too high and the loss diverges, too low and you wait days.
    float learning_rate = 3e-4f;

    // Standard deviation for random weight initialisation (GPT-2 uses 0.02).
    float initialization_standard_deviation = 0.02f;

    Model_Config() = default;

    // ────────────────────────────────
    // DERIVED VALUE — channels per attention head
    // Written as a function so it can never fall out of sync with the two values
    // it depends on.
    // ────────────────────────────────
    int get_head_dimension() const
    {
        return embedding_dimension / number_of_heads;
    }

    // Catch the classic setup mistake before it produces confusing shape errors
    // deep inside attention. Returns true if the configuration is usable.
    bool validate() const
    {
        if (embedding_dimension % number_of_heads != 0)
        {
            std::cout << "CONFIG ERROR: embedding_dimension ("
                      << embedding_dimension
                      << ") must divide evenly by number_of_heads ("
                      << number_of_heads << ")\n";
            return false;
        }

        if (vocabulary_size <= 0)
        {
            std::cout << "CONFIG ERROR: vocabulary_size was never set\n";
            return false;
        }

        return true;
    }

    void print() const
    {
        std::cout << "─── Model configuration ───\n";
        std::cout << "  vocabulary_size     : " << vocabulary_size << "\n";
        std::cout << "  block_size (context): " << block_size << "\n";
        std::cout << "  embedding_dimension : " << embedding_dimension << "\n";
        std::cout << "  number_of_heads     : " << number_of_heads
                  << "  (" << get_head_dimension() << " channels each)\n";
        std::cout << "  number_of_layers    : " << number_of_layers << "\n";
        std::cout << "  batch_size          : " << batch_size << "\n";
        std::cout << "  learning_rate       : " << learning_rate << "\n";
    }
};


#endif // __MODEL_CONFIG_H__
