/*
 * char_tokenizer.h — turns text into numbers and back again
 *
 * A language model cannot consume letters. It consumes integers, which it then
 * looks up in an embedding table. The tokenizer is the bridge.
 *
 * WHY CHARACTER LEVEL FIRST:
 * Real models use sub-word tokenizers (BPE, SentencePiece) with tens of
 * thousands of entries. Those are a whole project on their own, and they add
 * nothing to your understanding of the Transformer itself. Character level gives
 * us a vocabulary of about 65 symbols for Tiny Shakespeare — tiny output layer,
 * fast training, and zero preprocessing. The model has to work harder to learn
 * spelling, which is exactly what makes early training progress so visible: you
 * watch it discover words.
 *
 * We swap in a word or sub-word tokenizer later for TinyStories. Because the
 * rest of the model only ever sees integers, that swap touches nothing else.
 */

#ifndef __CHAR_TOKENIZER_H__
#define __CHAR_TOKENIZER_H__

#include <string>
#include <vector>


// ────────────────────────────────────────────────────────────────────────────────
// CHAR_TOKENIZER
// Builds a vocabulary from a body of text, then maps characters <-> integers.
// ────────────────────────────────────────────────────────────────────────────────
class Char_Tokenizer
{
private:
    // ────────────────────────────────
    // THE TWO LOOKUP TABLES
    //
    // index_to_character: token id -> the character it stands for.
    //                     Also defines the vocabulary size (its length).
    //
    // character_to_index: the reverse map. We use a fixed 256-entry array rather
    //                     than a std::map because a byte can only take 256
    //                     values, so an array IS the perfect hash table for this
    //                     job — one indexed read, no tree walking. Entries for
    //                     characters that never appear hold -1.
    // ────────────────────────────────

    std::vector<char> index_to_character;
    std::vector<int>  character_to_index;

public:
    Char_Tokenizer();

    // ────────────────────────────────
    // Scan the text, find every distinct character, and assign each one an id.
    //
    // Ids are handed out in ascending byte order (so '\n' gets 0, ' ' gets 1,
    // and so on). Sorted order matters: it makes the vocabulary deterministic,
    // so a model saved today still decodes correctly tomorrow.
    //
    // Returns the resulting vocabulary size.
    // ────────────────────────────────
    int build_vocabulary_from_text(const std::string& text);

    // How many distinct symbols the model must choose between. This becomes the
    // width of the model's output layer.
    int get_vocabulary_size() const;

    // Get the character behind a token id (used when printing generated text).
    char get_character_for_index(int index) const;

    // Text  ->  list of token ids. Unknown characters are skipped.
    std::vector<int> encode_text(const std::string& text) const;

    // List of token ids  ->  text.
    std::string decode_tokens(const std::vector<int>& tokens) const;

    // Print the vocabulary so you can see exactly what the model is working with.
    void print_vocabulary() const;
};


#endif // __CHAR_TOKENIZER_H__
