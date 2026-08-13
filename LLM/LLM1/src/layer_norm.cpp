/*
 * layer_norm.cpp — implementation of Layer Normalisation
 */

#include "layer_norm.h"

#include <cmath>


// ────────────────────────────────────────────────────────────────────────────────
// CONSTRUCTOR
// ────────────────────────────────────────────────────────────────────────────────
Layer_Norm::Layer_Norm()
    : feature_dimension(0),
      epsilon(1e-5f)    // the standard value used by essentially every framework
{
}


// ────────────────────────────────────────────────────────────────────────────────
// INITIALIZE
// ────────────────────────────────────────────────────────────────────────────────
void Layer_Norm::initialize(int dimension)
{
    feature_dimension = dimension;

    gain = Matrix(1, feature_dimension);
    bias = Matrix(1, feature_dimension);

    // Gain of 1 and bias of 0 means "pass the normalised values straight
    // through". The layer starts as pure normalisation and learns from there.
    gain.fill_with_ones();
    bias.fill_with_zeros();

    gradient_gain = Matrix(1, feature_dimension);
    gradient_bias = Matrix(1, feature_dimension);
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD
//
// One row (one token) at a time. Four steps per row: mean, variance, normalise,
// then scale and shift.
// ────────────────────────────────────────────────────────────────────────────────
Matrix Layer_Norm::compute_forward(const Matrix& input)
{
    int number_of_tokens = input.rows;

    Matrix output(number_of_tokens, feature_dimension);

    // Prepare the caches for the backward pass
    cached_normalized = Matrix(number_of_tokens, feature_dimension);
    cached_inverse_std.assign(number_of_tokens, 0.0f);

    for (int t = 0; t < number_of_tokens; ++t)
    {
        // ── Step 1: the mean of this token's channels ─────────────────────────
        float sum = 0.0f;

        for (int c = 0; c < feature_dimension; ++c)
        {
            sum += input.at(t, c);
        }

        float mean = sum / feature_dimension;

        // ── Step 2: the variance — average squared distance from the mean ─────
        float sum_of_squared_differences = 0.0f;

        for (int c = 0; c < feature_dimension; ++c)
        {
            float difference = input.at(t, c) - mean;
            sum_of_squared_differences += difference * difference;
        }

        float variance = sum_of_squared_differences / feature_dimension;

        // ── Step 3: normalise ─────────────────────────────────────────────────
        // We compute 1/sqrt(variance) once and multiply, rather than dividing
        // C times. Also cached, because backpropagation needs this exact value.
        float inverse_standard_deviation = 1.0f / std::sqrt(variance + epsilon);

        cached_inverse_std[t] = inverse_standard_deviation;

        for (int c = 0; c < feature_dimension; ++c)
        {
            float normalized_value =
                (input.at(t, c) - mean) * inverse_standard_deviation;

            cached_normalized.at(t, c) = normalized_value;

            // ── Step 4: scale and shift with the learnable parameters ─────────
            output.at(t, c) = normalized_value * gain.at(0, c) + bias.at(0, c);
        }
    }

    return output;
}


// ────────────────────────────────────────────────────────────────────────────────
// HOUSEKEEPING
// ────────────────────────────────────────────────────────────────────────────────
void Layer_Norm::zero_gradients()
{
    gradient_gain.fill_with_zeros();
    gradient_bias.fill_with_zeros();
}


int Layer_Norm::count_parameters() const
{
    // One gain and one bias per channel
    return 2 * feature_dimension;
}
