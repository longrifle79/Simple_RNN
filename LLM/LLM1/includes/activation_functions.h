/*
 * activation_functions.h — the non-linear pieces of the network
 *
 * This mirrors the Activation_Functions class from the LSTM project. There the
 * gates inherited from it so they could call sigmoid() directly; we do the same
 * here so Feed_Forward can call gelu() as if it were its own method.
 *
 * A Transformer needs two non-linearities:
 *   softmax — turns a row of raw scores into probabilities that sum to 1.
 *             Used twice: inside attention, and at the output over the vocabulary.
 *   GELU    — the activation inside the feed-forward block. Replaces the tanh
 *             and sigmoid we used in the LSTM.
 */

#ifndef __ACTIVATION_FUNCTIONS_H__
#define __ACTIVATION_FUNCTIONS_H__

#include "matrix.h"
#include <cmath>


// ────────────────────────────────────────────────────────────────────────────────
// ACTIVATION_FUNCTIONS
// A bag of small mathematical helpers. Layers inherit from it, exactly like the
// LSTM gates inherited sigmoid and tanh.
// ────────────────────────────────────────────────────────────────────────────────





class Activation_Functions
{
public:
    Activation_Functions() = default;

    // ────────────────────────────────
    // GELU — Gaussian Error Linear Unit
    //
    // The activation used by GPT-2, GPT-3 and most Transformers since.
    //
    // ReLU is a hard switch: negative in, exactly zero out. GELU is a soft
    // version of the same idea — it multiplies x by roughly "the probability
    // that x is worth keeping". Strongly negative inputs get squashed to near
    // zero, strongly positive inputs pass through almost untouched, and inputs
    // near zero get a gentle, smooth blend of the two.
    //
    // The formula below is the tanh approximation from the original paper. The
    // exact version needs the Gaussian error function; the approximation is
    // cheaper, and every major implementation uses it.
    //
    //   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    //
    // WHY IT MATTERS: without any non-linearity, stacking layers is pointless —
    // a chain of matrix multiplies collapses into one single matrix multiply.
    // The non-linearity is what makes depth buy you anything.
    // ────────────────────────────────



    float gelu(float x);



    // Derivative of GELU with respect to its input.
    // Not called during the forward pass; it is here ready for backpropagation.

    
    float gelu_derivative(float x);

    // ────────────────────────────────
    // SOFTMAX, applied independently to each row of a matrix
    //
    //   softmax(row)[i] = exp(row[i]) / sum over j of exp(row[j])
    //
    // Every output is positive and each row sums to exactly 1, so a row can be
    // read as a probability distribution.
    //
    // NUMERICAL STABILITY — the single most important detail in this file:
    // exp(1000) overflows a float to infinity and the whole row becomes NaN.
    // The fix is to subtract the row's maximum before exponentiating. Since
    //     exp(a - m) / sum(exp(b - m))  ==  exp(a) / sum(exp(b))
    // the answer is mathematically identical, but now the largest exponent is
    // exp(0) = 1 and nothing can blow up. Never write softmax without this.
    //
    // Modifies the matrix in place.
    // ────────────────────────────────


    void softmax_rows_in_place(Matrix& matrix);

    // ────────────────────────────────
    // Softmax over just the first `count` entries of one row.
    //
    // Causal attention needs this: at position t only the first t+1 scores are
    // real, and the rest refer to future tokens the model is not allowed to see.
    // Rather than filling them with a large negative number and exponentiating
    // anyway, we simply never look at them and write 0 into those slots.
    // ────────────────────────────────


    void softmax_row_prefix_in_place(Matrix& matrix, int row, int count);
};


#endif // __ACTIVATION_FUNCTIONS_H__
