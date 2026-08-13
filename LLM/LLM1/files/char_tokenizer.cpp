/*
 * char_tokenizer.cpp — implementation of the character-level tokenizer
 */

#include "char_tokenizer.h"

#include <iostream>


// ────────────────────────────────────────────────────────────────────────────────
// CONSTRUCTOR
// Start with an empty vocabulary and a reverse table full of -1 ("not present").
// ────────────────────────────────────────────────────────────────────────────────
Char_Tokenizer::Char_Tokenizer()
    : character_to_index(256, -1)   // 256 slots, one per possible byte value
{
    // index_to_character starts empty; build_vocabulary_from_text fills it.
}


// ────────────────────────────────────────────────────────────────────────────────
// BUILD VOCABULARY
// ────────────────────────────────────────────────────────────────────────────────
int Char_Tokenizer::build_vocabulary_from_text(const std::string& text)
{
    // Start clean in case this is called twice.
    index_to_character.clear();
    character_to_index.assign(256, -1);

    // ── Pass 1: mark which byte values actually occur ─────────────────────────
    std::vector<bool> character_is_present(256, false);

    for (size_t i = 0; i < text.size(); ++i)
    {
        // IMPORTANT: char may be signed on some platforms, so a byte above 127
        // would come out negative and index out of bounds. Casting to unsigned
        // char first guarantees a value in [0, 255].
        unsigned char byte_value = static_cast<unsigned char>(text[i]);

        character_is_present[byte_value] = true;
    }

    // ── Pass 2: hand out ids in ascending byte order ──────────────────────────
    // Walking 0..255 in order is what makes the vocabulary deterministic.
    for (int byte_value = 0; byte_value < 256; ++byte_value)
    {
        if (character_is_present[byte_value])
        {
            // This character's id is simply "how many we have assigned so far"
            int new_index = static_cast<int>(index_to_character.size());

            index_to_character.push_back(static_cast<char>(byte_value));
            character_to_index[byte_value] = new_index;
        }
    }

    return get_vocabulary_size();
}


// ────────────────────────────────────────────────────────────────────────────────
// SIMPLE ACCESSORS
// ────────────────────────────────────────────────────────────────────────────────
int Char_Tokenizer::get_vocabulary_size() const
{
    return static_cast<int>(index_to_character.size());
}


char Char_Tokenizer::get_character_for_index(int index) const
{
    // Out-of-range ids should never happen, but returning '?' beats crashing
    // while you are experimenting with sampling code.
    if (index < 0 || index >= get_vocabulary_size())
    {
        return '?';
    }

    return index_to_character[index];
}


// ────────────────────────────────────────────────────────────────────────────────
// ENCODE — text to token ids
// ────────────────────────────────────────────────────────────────────────────────
std::vector<int> Char_Tokenizer::encode_text(const std::string& text) const
{
    std::vector<int> tokens;

    // Reserving up front avoids repeated reallocation on a million-character file
    tokens.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i)
    {
        unsigned char byte_value = static_cast<unsigned char>(text[i]);

        int index = character_to_index[byte_value];

        // -1 means this character was not in the training text, so we have no id
        // for it. We silently drop it. A production tokenizer would map it to a
        // dedicated <unknown> token instead.
        if (index >= 0)
        {
            tokens.push_back(index);
        }
    }

    return tokens;
}


// ────────────────────────────────────────────────────────────────────────────────
// DECODE — token ids back to text
// ────────────────────────────────────────────────────────────────────────────────
std::string Char_Tokenizer::decode_tokens(const std::vector<int>& tokens) const
{
    std::string text;
    text.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i)
    {
        text.push_back(get_character_for_index(tokens[i]));
    }

    return text;
}


// ────────────────────────────────────────────────────────────────────────────────
// PRINT VOCABULARY — so you can see exactly what the model can produce
// ────────────────────────────────────────────────────────────────────────────────
void Char_Tokenizer::print_vocabulary() const
{
    std::cout << "Vocabulary (" << get_vocabulary_size() << " symbols): ";

    for (int i = 0; i < get_vocabulary_size(); ++i)
    {
        char c = index_to_character[i];

        // Whitespace would be invisible, so name it instead of printing it
        if (c == '\n')
        {
            std::cout << "\\n";
        }
        else if (c == ' ')
        {
            std::cout << "_";
        }
        else
        {
            std::cout << c;
        }
    }

    std::cout << "\n";
}
