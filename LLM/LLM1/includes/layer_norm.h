/*
 * layer_norm.h — Layer Normalisation
 *
 * WHAT IT DOES:
 * For each token independently, take that token's vector of C numbers, and
 * rescale it so those C numbers have mean 0 and standard deviation 1. Then
 * multiply by a learnable gain and add a learnable bias.
 *
 *     normalised = (x - mean) / sqrt(variance + epsilon)
 *     output     = normalised * gain + bias
 *
 * WHY IT IS THERE:
 * Deep networks drift. After a few layers of matrix multiplies and additions,
 * activations can grow enormous or collapse toward zero, and either way learning
 * stalls. LayerNorm resets the scale at every stage, so layer 4 receives inputs
 * in the same range layer 1 did. It is the main reason a Transformer can be
 * stacked dozens of layers deep and still train.
 *
 * WHY EACH TOKEN SEPARATELY (and not across the batch, like BatchNorm):
 * Because it makes each token's result independent of whatever else happened to
 * be in the batch. That matters enormously at generation time, when the batch is
 * a single sequence. LayerNorm behaves identically whether you feed it one
 * sequence or a thousand.
 *
 * WHY GAIN AND BIAS EXIST:
 * Forcing mean 0 and variance 1 is a strong constraint, and sometimes the layer
 * genuinely wants a different scale. The gain (initialised to 1) and bias
 * (initialised to 0) let the network undo the normalisation if it needs to, so
 * we get the stability benefit without permanently removing a degree of freedom.
 */

#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "matrix.h"

#include <vector>


// ────────────────────────────────────────────────────────────────────────────────
// LAYER_NORM
// ────────────────────────────────────────────────────────────────────────────────
class Layer_Norm
{
public:
    int feature_dimension;      // C — the number of channels per token

    // ────────────────────────────────
    // LEARNABLE PARAMETERS — one number per channel, shared by all tokens
    // ────────────────────────────────
    Matrix gain;    // [1, C], starts at all ones  (often written gamma)
    Matrix bias;    // [1, C], starts at all zeros (often written beta)

    Matrix gradient_gain;
    Matrix gradient_bias;

    // ────────────────────────────────
    // CACHES FOR BACKPROPAGATION
    //
    // The backward pass through LayerNorm is the fiddliest derivative in the
    // model, because changing one element of x changes the mean and the variance,
    // which changes every other output. Saving the normalised values and the
    // inverse standard deviation now makes that derivation far more manageable.
    // ────────────────────────────────
    Matrix cached_normalized;               // [T, C], the values before gain/bias
    std::vector<float> cached_inverse_std;  // one per token

    // A tiny constant added to the variance before taking the square root, so a
    // token whose channels happen to be identical does not divide by zero.
    float epsilon;

    Layer_Norm();

    void initialize(int dimension);

    // Forward pass over a [T, C] matrix. Each row is normalised on its own.
    Matrix compute_forward(const Matrix& input);

    void zero_gradients();

    int count_parameters() const;
};


#endif // __LAYER_NORM_H__
