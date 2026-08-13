/*
 * main.cpp — Stage 1: verify the forward pass
 *
 * A decoder-only Transformer (GPT-style) language model in plain C++17, written
 * the same way the LSTM unit was: no external libraries, every operation
 * visible, clarity ahead of speed.
 *
 * WHAT THIS STAGE DOES:
 *   - loads Tiny Shakespeare and builds a character-level vocabulary
 *   - checks the tokenizer round-trips text correctly
 *   - builds the model and reports where its parameters live
 *   - runs one batch through the forward pass and computes the loss
 *   - checks that loss against log(vocabulary_size)
 *
 * THE CHECK THAT MATTERS:
 * An untrained model knows nothing, so it should spread its probability evenly
 * across all V possible next tokens. That gives a loss of exactly log(V) — about
 * 4.17 for a 65-symbol vocabulary. If the first loss comes out near that number,
 * then the embeddings, all four attention projections, both LayerNorms, the
 * feed-forward blocks, the residual connections and the output head are all
 * wired up correctly and producing sensibly scaled values.
 *
 * If instead it comes out at 8, or 0.5, or NaN, something is wrong and it is far
 * better to find that now than after an hour of training that goes nowhere. This
 * is the same instinct as the numerical gradient check on the RNN project.
 *
 * BUILD (CMake):
 *   mkdir -p build && cd build && cmake .. && make
 *   ./main
 *
 * BUILD (one line, no CMake):
 *   see README.md — the command uses shell globs, and a glob written inside a
 *   C block comment contains the character pair that opens a comment, which
 *   makes the compiler warn. A small reminder that comments are parsed too.
 *
 * DATA:
 *   curl -o data/tinyshakespeare.txt \
 *     https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>

#include "matrix.h"
#include "text_dataset.h"
#include "model_config.h"
#include "gpt_model.h"


// ─── Where the data lives (edit this, or pass it on the command line) ──────────
//
// Two paths are tried in order. The second one exists because a CMake build puts
// the executable in build/, one level below the data. Rather than remembering
// which directory to run from, we just look in both.
std::string DATA_FILE     = "data/tinyshakespeare.txt";
std::string DATA_FILE_ALT = "../data/tinyshakespeare.txt";

// Fraction of the corpus used for training; the rest is held out.
const float TRAIN_FRACTION = 0.9f;

// A small piece of text used only if the dataset file cannot be found, so the
// program still demonstrates itself before you have downloaded anything.
const char* FALLBACK_TEXT =
    "First Citizen:\n"
    "Before we proceed any further, hear me speak.\n"
    "\n"
    "All:\n"
    "Speak, speak.\n"
    "\n"
    "First Citizen:\n"
    "You are all resolved rather to die than to famish?\n"
    "\n"
    "All:\n"
    "Resolved. resolved.\n";


int main(int argc, char** argv)
{
    // Allow the data path to be overridden: ./main path/to/text.txt
    if (argc > 1)
    {
        DATA_FILE = argv[1];
    }

    // Fixed seed so every run of this program produces identical numbers.
    // Change it to see how much the initial loss varies (it should barely move).
    set_random_seed(1234);

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  Transformer LLM — Stage 1: forward pass verification\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";


    // ────────────────────────────────────────────────────────────────────────
    // PART 1 — LOAD THE DATA
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── Loading data ───\n";

    Text_Dataset dataset;

    if (dataset.load_from_file(DATA_FILE, TRAIN_FRACTION))
    {
        std::cout << "Loaded: " << DATA_FILE << "\n";
    }
    else if (dataset.load_from_file(DATA_FILE_ALT, TRAIN_FRACTION))
    {
        // Found it one directory up — we are probably running from build/
        std::cout << "Loaded: " << DATA_FILE_ALT << "\n";
    }
    else
    {
        // Not fatal — fall back so the rest of the program still runs.
        std::cout << "Could not open " << DATA_FILE << "\n";
        std::cout << "Using the small built-in sample instead.\n";
        std::cout << "Download the real dataset with:\n"
                  << "  curl -o data/tinyshakespeare.txt \\\n"
                  << "    https://raw.githubusercontent.com/karpathy/char-rnn/"
                     "master/data/tinyshakespeare/input.txt\n";

        dataset.load_from_string(FALLBACK_TEXT, TRAIN_FRACTION);
    }

    dataset.print_summary();
    dataset.tokenizer.print_vocabulary();
    std::cout << "\n";


    // ────────────────────────────────────────────────────────────────────────
    // PART 2 — CHECK THE TOKENIZER ROUND-TRIPS
    //
    // encode then decode must give back exactly what went in. If this fails,
    // nothing downstream can possibly be right, so we test it first.
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── Tokenizer check ───\n";

    std::string sample_text = "First Citizen:";

    std::vector<int> encoded = dataset.tokenizer.encode_text(sample_text);

    std::string decoded = dataset.tokenizer.decode_tokens(encoded);

    std::cout << "Original : \"" << sample_text << "\"\n";

    std::cout << "Encoded  : ";
    for (size_t i = 0; i < encoded.size(); ++i)
    {
        std::cout << encoded[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Decoded  : \"" << decoded << "\"\n";
    std::cout << "Round trip: " << (decoded == sample_text ? "PASS" : "FAIL") << "\n\n";


    // ────────────────────────────────────────────────────────────────────────
    // PART 3 — LOOK AT ONE TRAINING EXAMPLE
    //
    // Printing the input/target pairs is the fastest way to convince yourself
    // the "predict the next token" setup is what you think it is.
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── What one training example looks like ───\n";

    std::vector<std::vector<int>> demo_inputs;
    std::vector<std::vector<int>> demo_targets;

    // One sequence, eight tokens, just to look at
    dataset.get_random_batch(1, 8, false, demo_inputs, demo_targets);

    for (int t = 0; t < 8; ++t)
    {
        // Show the context so far and the token that should follow it
        std::vector<int> context(demo_inputs[0].begin(),
                                 demo_inputs[0].begin() + t + 1);

        std::cout << "  after \""
                  << dataset.tokenizer.decode_tokens(context)
                  << "\" -> predict '"
                  << dataset.tokenizer.get_character_for_index(demo_targets[0][t])
                  << "'\n";
    }
    std::cout << "\n";


    // ────────────────────────────────────────────────────────────────────────
    // PART 4 — BUILD THE MODEL
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── Building the model ───\n";

    Model_Config config;

    // Everything else keeps its default from model_config.h; only the
    // vocabulary size has to come from the data.
    config.vocabulary_size = dataset.tokenizer.get_vocabulary_size();

    if (!config.validate())
    {
        return 1;   // validate() already explained what was wrong
    }

    config.print();
    std::cout << "\n";

    Gpt_Model model;
    model.initialize(config);

    model.print_parameter_breakdown();
    std::cout << "\n";


    // ────────────────────────────────────────────────────────────────────────
    // PART 5 — ONE FORWARD PASS
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── Forward pass on a single sequence ───\n";

    std::vector<std::vector<int>> input_batch;
    std::vector<std::vector<int>> target_batch;

    dataset.get_random_batch(config.batch_size,
                             config.block_size,
                             false,
                             input_batch,
                             target_batch);

    Matrix logits = model.compute_forward(input_batch[0]);

    std::cout << "Input tokens : " << input_batch[0].size() << "\n";
    std::cout << "Logits shape : [" << logits.rows << " x " << logits.cols
              << "]   (one row per position, one column per vocabulary symbol)\n\n";

    logits.print("First few logits", 4, 8);

    float single_sequence_loss =
        model.compute_cross_entropy_loss(logits, target_batch[0]);

    std::cout << "Loss on this sequence: " << single_sequence_loss << "\n\n";


    // ────────────────────────────────────────────────────────────────────────
    // PART 6 — THE VERIFICATION THAT MATTERS
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── Checking the initial loss ───\n";

    float batch_loss = model.compute_batch_loss(input_batch, target_batch);

    float expected_loss = std::log(static_cast<float>(config.vocabulary_size));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Measured loss over " << config.batch_size
              << " sequences : " << batch_loss << "\n";
    std::cout << "Expected  log(" << config.vocabulary_size
              << ")                  : " << expected_loss << "\n";

    float difference = std::fabs(batch_loss - expected_loss);

    std::cout << "Difference                        : " << difference << "\n\n";

    // A tolerance of 0.5 is generous but meaningful. A correct untrained model
    // lands within a few hundredths; a broken one is nowhere near.
    if (difference < 0.5f)
    {
        std::cout << "PASS — the untrained model is close to uniform, exactly as\n"
                  << "it should be. Embeddings, attention, layer norms,\n"
                  << "feed-forward blocks, residuals and the output head are all\n"
                  << "wired up correctly.\n\n";
    }
    else
    {
        std::cout << "FAIL — the initial loss is too far from log(V). Something in\n"
                  << "the forward pass is wrong. Check the initialisation standard\n"
                  << "deviation first, then the layer norms.\n\n";
    }


    // ────────────────────────────────────────────────────────────────────────
    // PART 7 — LOOK AT AN ATTENTION PATTERN
    //
    // Untrained, these weights are nearly uniform over the visible positions,
    // which is itself informative: note the strict lower-triangular shape. Row 0
    // has one non-zero entry, row 1 has two, and so on. That triangle IS the
    // causal mask. Nothing above the diagonal, ever.
    // ────────────────────────────────────────────────────────────────────────
    std::cout << "─── Attention weights, block 0, head 0 ───\n";
    std::cout << "(row t = where token t looked; note the lower triangle)\n\n";

    const Matrix& attention =
        model.transformerBlocks[0].multiHeadAttention.cached_attention_weights[0];

    attention.print("attention[block 0][head 0]", 6, 6);

    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  Stage 1 complete. Next: the backward pass.\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";

    return 0;
}
