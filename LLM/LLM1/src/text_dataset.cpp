/*
 * text_dataset.cpp — implementation of dataset loading and batching
 */

#include "text_dataset.h"
#include "matrix.h"      // for get_random_engine()

#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cassert>


// ────────────────────────────────────────────────────────────────────────────────
// LOAD FROM FILE
// ────────────────────────────────────────────────────────────────────────────────
bool Text_Dataset::load_from_file(const std::string& file_path, float train_fraction)
{
    // Open in binary mode so no newline translation happens behind our back.
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open())
    {
        return false;   // caller decides what to do about it
    }

    // Read the whole file into a string in one shot. The stringstream buffer
    // trick is the shortest correct way to slurp a file in C++.
    std::stringstream buffer;
    buffer << file.rdbuf();

    file.close();

    load_from_string(buffer.str(), train_fraction);

    return true;
}


// ────────────────────────────────────────────────────────────────────────────────
// LOAD FROM STRING — where the real work happens
// ────────────────────────────────────────────────────────────────────────────────
void Text_Dataset::load_from_string(const std::string& text, float train_fraction)
{
    // ── Step 1: work out the alphabet ─────────────────────────────────────────
    tokenizer.build_vocabulary_from_text(text);

    // ── Step 2: turn the entire corpus into integers ──────────────────────────
    std::vector<int> all_tokens = tokenizer.encode_text(text);

    // ── Step 3: split into training and validation, by position ───────────────
    size_t split_point = static_cast<size_t>(all_tokens.size() * train_fraction);

    training_tokens.assign(all_tokens.begin(),
                           all_tokens.begin() + split_point);

    validation_tokens.assign(all_tokens.begin() + split_point,
                             all_tokens.end());
}


// ────────────────────────────────────────────────────────────────────────────────
// GET RANDOM BATCH
// ────────────────────────────────────────────────────────────────────────────────
void Text_Dataset::get_random_batch(int batch_size,
                                    int block_size,
                                    bool use_validation_set,
                                    std::vector<std::vector<int>>& input_batch,
                                    std::vector<std::vector<int>>& target_batch) const
{
    // Pick which pool of tokens to sample from.
    const std::vector<int>& source_tokens =
        use_validation_set ? validation_tokens : training_tokens;

    // We need block_size input tokens PLUS one more for the final target, so the
    // last legal starting index is size - block_size - 1.
    assert(static_cast<int>(source_tokens.size()) > block_size + 1);

    int highest_legal_start =
        static_cast<int>(source_tokens.size()) - block_size - 1;

    // A uniform distribution over every valid starting position.
    std::uniform_int_distribution<int> start_position(0, highest_legal_start);

    std::mt19937& engine = get_random_engine();

    // Size the output containers: batch_size rows of block_size tokens each.
    input_batch.assign(batch_size, std::vector<int>(block_size));
    target_batch.assign(batch_size, std::vector<int>(block_size));

    for (int b = 0; b < batch_size; ++b)
    {
        // Each sequence in the batch starts at its own random offset.
        int start = start_position(engine);

        for (int t = 0; t < block_size; ++t)
        {
            // The input is the window starting at `start`
            input_batch[b][t] = source_tokens[start + t];

            // The target is that same window shifted right by one. This is the
            // "predict the next token" objective, made concrete.
            target_batch[b][t] = source_tokens[start + t + 1];
        }
    }
}


// ────────────────────────────────────────────────────────────────────────────────
// PRINT SUMMARY
// ────────────────────────────────────────────────────────────────────────────────
void Text_Dataset::print_summary() const
{
    std::cout << "Training tokens:   " << training_tokens.size() << "\n";
    std::cout << "Validation tokens: " << validation_tokens.size() << "\n";
    std::cout << "Vocabulary size:   " << tokenizer.get_vocabulary_size() << "\n";
}
