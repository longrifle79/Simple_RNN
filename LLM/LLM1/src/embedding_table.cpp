/*
 * embedding_table.cpp — implementation of the embedding lookup table
 */

#include "embedding_table.h"

#include <cassert>


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE
// ────────────────────────────────────────────────────────────────────────────────
void Embedding_Table::initialize(int entries, int dimension, float init_standard_deviation)
{
    number_of_entries   = entries;
    embedding_dimension = dimension;

    table = Matrix(number_of_entries, embedding_dimension);

    // Random start, exactly like any other weight matrix. There is nothing
    // special about an embedding — it is a weight matrix we happen to index into
    // rather than multiply by.
    table.fill_with_random_normal(0.0f, init_standard_deviation);

    gradient_table = Matrix(number_of_entries, embedding_dimension);
}


// ────────────────────────────────────────────────────────────────────────────────
// LOOKUP ROWS
// ────────────────────────────────────────────────────────────────────────────────
Matrix Embedding_Table::lookup_rows(const std::vector<int>& indices)
{
    // Remember which rows we touched, for the backward pass.
    cached_indices = indices;

    int number_of_tokens = static_cast<int>(indices.size());

    Matrix result(number_of_tokens, embedding_dimension);

    for (int t = 0; t < number_of_tokens; ++t)
    {
        int row_to_copy = indices[t];

        // A bad index means the tokenizer and the model disagree about the
        // vocabulary size — worth catching loudly rather than reading garbage.
        assert(row_to_copy >= 0 && row_to_copy < number_of_entries);

        // Copy the whole row across
        for (int c = 0; c < embedding_dimension; ++c)
        {
            result.at(t, c) = table.at(row_to_copy, c);
        }
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// MAKE POSITION INDICES — [0, 1, 2, ..., count-1]
// ────────────────────────────────────────────────────────────────────────────────
std::vector<int> Embedding_Table::make_position_indices(int count)
{
    std::vector<int> positions(count);

    for (int i = 0; i < count; ++i)
    {
        positions[i] = i;
    }

    return positions;
}


// ────────────────────────────────────────────────────────────────────────────────
// HOUSEKEEPING
// ────────────────────────────────────────────────────────────────────────────────
void Embedding_Table::zero_gradients()
{
    gradient_table.fill_with_zeros();
}


int Embedding_Table::count_parameters() const
{
    return number_of_entries * embedding_dimension;
}
