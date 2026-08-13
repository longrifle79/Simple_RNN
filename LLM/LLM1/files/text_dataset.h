/*
 * text_dataset.h — loads a text file and serves random training batches
 *
 * THE TASK A LANGUAGE MODEL LEARNS:
 * Given the tokens seen so far, predict the next one. That is all it ever does.
 *
 * So a training example is a pair of sequences offset by exactly one position:
 *
 *     tokens:  [F][i][r][s][t][ ][C][i]...
 *     input:   [F][i][r][s][t][ ][C]        (block_size = 7 tokens)
 *     target:     [i][r][s][t][ ][C][i]     (the same window, shifted by one)
 *
 * Position 0 of the input must predict position 0 of the target, position 1 must
 * predict position 1, and so on. One window of length T therefore gives us T
 * separate prediction problems, not one — which is why Transformers train so
 * much faster than the RNN did. In the RNN we had to walk the sequence one step
 * at a time; here all T predictions happen in a single parallel forward pass.
 */

#ifndef __TEXT_DATASET_H__
#define __TEXT_DATASET_H__

#include "char_tokenizer.h"

#include <string>
#include <vector>


// ────────────────────────────────────────────────────────────────────────────────
// TEXT_DATASET
// Owns the tokenizer, the encoded corpus, and the train/validation split.
// ────────────────────────────────────────────────────────────────────────────────
class Text_Dataset
{
public:
    // Public so main() can ask it to decode generated text later.
    Char_Tokenizer tokenizer;

    // ────────────────────────────────
    // THE CORPUS, ALREADY ENCODED
    //
    // We store the entire dataset as one long list of token ids. Tiny
    // Shakespeare is about 1.1 million characters, so this is ~4 MB as ints —
    // nothing. There is no need for streaming or lazy loading at this scale.
    // ────────────────────────────────

    std::vector<int> training_tokens;
    std::vector<int> validation_tokens;

    Text_Dataset() = default;

    // ────────────────────────────────
    // Read a text file, build the vocabulary from it, encode it, and split it.
    //
    // train_fraction of 0.9 means the last 10% of the text is held out. We split
    // by position rather than randomly because the data is one continuous
    // document: a random split would leak neighbouring text across the boundary
    // and make validation loss look better than it really is.
    //
    // Returns true on success, false if the file could not be opened.
    // ────────────────────────────────
    bool load_from_file(const std::string& file_path, float train_fraction);

    // Same thing but from a string already in memory. Used as a fallback in
    // main() so the program still runs before you have downloaded the dataset.
    void load_from_string(const std::string& text, float train_fraction);

    // ────────────────────────────────
    // Draw one random batch.
    //
    // batch_size independent windows are chosen from random starting points.
    // Each window is block_size tokens long.
    //
    // input_batch  ends up as batch_size rows, each block_size token ids long
    // target_batch is the same windows shifted one position to the right
    //
    // WHY RANDOM STARTS: sampling random windows every step means the model sees
    // an enormous variety of contexts and never memorises the order of the file.
    // It is also far simpler than shuffling and tracking epochs.
    // ────────────────────────────────
    void get_random_batch(int batch_size,
                          int block_size,
                          bool use_validation_set,
                          std::vector<std::vector<int>>& input_batch,
                          std::vector<std::vector<int>>& target_batch) const;

    // Convenience printers used by main() to sanity check the pipeline
    void print_summary() const;
};


#endif // __TEXT_DATASET_H__
