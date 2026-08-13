/*
 * embedding_table.h — a learnable lookup table of vectors
 *
 * WHAT AN EMBEDDING IS:
 * The tokenizer gives us integers, but the integer 42 has no meaning — it is not
 * "twice as much" as 21. So we keep a table with one row per possible token, and
 * each row is a vector of embedding_dimension numbers. Token 42 means "row 42".
 *
 * Those rows start as random noise and are learned by gradient descent along
 * with everything else. After training, characters that behave similarly end up
 * with similar vectors, because the network had no reason to keep them apart.
 *
 * WE USE THIS CLASS TWICE, FOR TWO DIFFERENT JOBS:
 *
 *   1. TOKEN EMBEDDING   — one row per vocabulary symbol. "What is this token?"
 *
 *   2. POSITION EMBEDDING — one row per slot in the context window.
 *                           "Where in the sequence am I?"
 *
 * Position embeddings are essential and easy to overlook. Attention is a
 * weighted sum over all previous tokens, and a sum does not care about order:
 * without positional information, "dog bites man" and "man bites dog" would look
 * identical to the model. The RNN got order for free because it walked the
 * sequence in time. A Transformer sees everything at once, so we must hand it
 * position explicitly, by adding a learned per-slot vector to each token vector.
 */

#ifndef __EMBEDDING_TABLE_H__
#define __EMBEDDING_TABLE_H__

#include "matrix.h"

#include <vector>


// ────────────────────────────────────────────────────────────────────────────────
// EMBEDDING_TABLE
// ────────────────────────────────────────────────────────────────────────────────
class Embedding_Table
{
public:
    int number_of_entries;      // vocabulary size, or context length
    int embedding_dimension;    // width of each row

    // The table itself: [number_of_entries, embedding_dimension]
    Matrix table;

    // Gradient buffer, same shape. Filled during backpropagation.
    Matrix gradient_table;

    // Which rows were used in the last forward pass. Backpropagation only needs
    // to touch those rows — in a batch of 32 sequences of 64 characters we look
    // at 2048 rows out of maybe 65, so remembering the ids saves nothing here,
    // but it will matter enormously once the vocabulary is 50,000 entries.
    std::vector<int> cached_indices;

    Embedding_Table() = default;

    void initialize(int entries, int dimension, float init_standard_deviation);

    // ────────────────────────────────
    // LOOKUP — the entire forward pass of an embedding layer
    //
    // Given a list of T indices, return a [T, embedding_dimension] matrix whose
    // row t is table row indices[t].
    //
    // Note there is no arithmetic here at all: this is a copy, not a multiply.
    // Mathematically it is equivalent to multiplying a one-hot vector by the
    // table, but doing it as a lookup is thousands of times faster and much
    // easier to read.
    // ────────────────────────────────
    Matrix lookup_rows(const std::vector<int>& indices);

    // Build a list of consecutive positions 0, 1, 2, ..., count-1.
    // Convenience for the position embedding, whose "indices" are always just
    // the slot numbers.
    static std::vector<int> make_position_indices(int count);

    void zero_gradients();

    int count_parameters() const;
};


#endif // __EMBEDDING_TABLE_H__
