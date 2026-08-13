/*
 * activation_functions.cpp — implementation of GELU and softmax
 */

#include "activation_functions.h"


// ────────────────────────────────────────────────────────────────────────────────
// GELU — tanh approximation
// ────────────────────────────────────────────────────────────────────────────────
float Activation_Functions::gelu(float x)
{
    // sqrt(2 / pi), precomputed so we do not recompute it millions of times
    const float sqrt_2_over_pi = 0.7978845608028654f;

    // The cubic correction term from the paper
    const float cubic_coefficient = 0.044715f;

    // Inside of the tanh
    float inner = sqrt_2_over_pi * (x + cubic_coefficient * x * x * x);

    // 0.5 * x * (1 + tanh(inner))
    // When inner is very negative, tanh -> -1, so the result -> 0 (input killed).
    // When inner is very positive, tanh -> +1, so the result -> x (input kept).
    return 0.5f * x * (1.0f + std::tanh(inner));
}


// ────────────────────────────────────────────────────────────────────────────────
// GELU DERIVATIVE
//
// Product rule applied to  0.5 * x * (1 + tanh(u))  where
//     u = sqrt(2/pi) * (x + 0.044715 * x^3)
//     du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
//
//     d/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * du/dx
//
// The (1 - tanh^2) piece is just the derivative of tanh, the same one we used in
// the LSTM project.
// ────────────────────────────────────────────────────────────────────────────────
float Activation_Functions::gelu_derivative(float x)
{
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float cubic_coefficient = 0.044715f;

    float inner = sqrt_2_over_pi * (x + cubic_coefficient * x * x * x);

    float tanh_inner = std::tanh(inner);

    // Derivative of the inside with respect to x
    float inner_derivative = sqrt_2_over_pi * (1.0f + 3.0f * cubic_coefficient * x * x);

    // First term: the "0.5 * (1 + tanh(u))" part differentiated as a constant
    // multiple of x. Second term: x times the derivative of the tanh factor.
    return 0.5f * (1.0f + tanh_inner)
         + 0.5f * x * (1.0f - tanh_inner * tanh_inner) * inner_derivative;
}


// ────────────────────────────────────────────────────────────────────────────────
// SOFTMAX over every row
// ────────────────────────────────────────────────────────────────────────────────
void Activation_Functions::softmax_rows_in_place(Matrix& matrix)
{
    for (int row = 0; row < matrix.rows; ++row)
    {
        // Every row uses its full width, so just reuse the prefix version.
        softmax_row_prefix_in_place(matrix, row, matrix.cols);
    }
}


// ────────────────────────────────────────────────────────────────────────────────
// SOFTMAX over the first `count` entries of a single row
// ────────────────────────────────────────────────────────────────────────────────
void Activation_Functions::softmax_row_prefix_in_place(Matrix& matrix, int row, int count)
{
    // ── Step 1: find the largest value in the active part of the row ──────────
    // This is the numerical-stability trick described in the header.
    float max_value = matrix.at(row, 0);

    for (int j = 1; j < count; ++j)
    {
        if (matrix.at(row, j) > max_value)
        {
            max_value = matrix.at(row, j);
        }
    }

    // ── Step 2: exponentiate (value - max) and total it up ────────────────────
    float sum_of_exponentials = 0.0f;

    for (int j = 0; j < count; ++j)
    {
        // Largest possible argument is now 0, so exp() can never overflow.
        float exponential = std::exp(matrix.at(row, j) - max_value);

        matrix.at(row, j) = exponential;   // stash it, we divide in step 3
        sum_of_exponentials += exponential;
    }

    // ── Step 3: divide each entry by the total so the row sums to 1 ───────────
    for (int j = 0; j < count; ++j)
    {
        matrix.at(row, j) = matrix.at(row, j) / sum_of_exponentials;
    }

    // ── Step 4: zero out the masked tail ──────────────────────────────────────
    // These positions are future tokens. They must contribute nothing, and
    // writing 0 makes that explicit rather than leaving stale values behind.
    for (int j = count; j < matrix.cols; ++j)
    {
        matrix.at(row, j) = 0.0f;
    }
}
