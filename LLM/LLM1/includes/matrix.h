/*
 * matrix.h — A small, deliberately simple matrix class
 *
 * WHY WE WRITE OUR OWN INSTEAD OF USING EIGEN:
 * In the SimpleRNN project we leaned on Eigen. Here we do it by hand for the
 * same reason we wrote the LSTM gates by hand — every multiply and every sum
 * should be visible. There is nowhere for the math to hide. If you later want
 * speed, you can swap the internals of matrix_multiply() for Eigen or BLAS
 * without touching a single line of the Transformer code.
 *
 * DESIGN:
 * A Matrix is nothing but a flat std::vector<float> plus two integers telling
 * us how to interpret it. Element (row, col) lives at index (row * cols + col).
 * This is called ROW-MAJOR order: row 0 is stored first, then row 1, and so on.
 *
 * SHAPE CONVENTION USED THROUGHOUT THE WHOLE PROJECT:
 * Almost every matrix in the Transformer has shape [T, C]:
 *     T = number of tokens in the sequence (time steps / context positions)
 *     C = number of channels (the embedding dimension)
 * So row t is "the vector that represents token t at this point in the network".
 * Keeping that one picture in your head makes the rest of the code readable.
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>       // std::vector — our flat storage
#include <string>       // std::string — used for print labels
#include <random>       // std::mt19937 — random weight initialisation


// ────────────────────────────────────────────────────────────────────────────────
// MATRIX CLASS
// A 2D block of floats. No expression templates, no operator overloading tricks,
// no clever memory games. Just numbers and two dimensions.
// ────────────────────────────────────────────────────────────────────────────────
class Matrix
{
public:
    // ────────────────────────────────
    // MEMBER VARIABLES
    // Left public on purpose. Getters and setters around a raw number grid would
    // add noise without adding safety, and you will constantly want to read
    // .rows and .cols while reasoning about shapes.
    // ────────────────────────────────

    int rows;                    // How many rows (usually T = sequence length)
    int cols;                    // How many columns (usually C = embedding dim)
    std::vector<float> values;   // The actual numbers, flattened row by row

    // ────────────────────────────────
    // CONSTRUCTORS
    // ────────────────────────────────

    // Empty matrix — 0x0. Needed so classes can hold Matrix members and size
    // them later inside their initialize() function.
    Matrix();

    // Build a matrix of the given size, filled with zeros.
    Matrix(int number_of_rows, int number_of_cols);

    // ────────────────────────────────
    // ELEMENT ACCESS
    // at(r, c) does the row-major index arithmetic for you.
    // Two versions: one that lets you write, one that only lets you read
    // (the const version, used when the matrix is passed as const&).
    // ────────────────────────────────

    float&       at(int row, int col);
    const float& at(int row, int col) const;

    // ────────────────────────────────
    // FILLING
    // ────────────────────────────────

    // Set every element to 0.0f. Used to reset gradient buffers each step.
    void fill_with_zeros();

    // Set every element to 1.0f. Used for LayerNorm's gain parameter.
    void fill_with_ones();

    // Fill with random numbers drawn from a normal (Gaussian) distribution.
    // Small random values break symmetry so neurons learn different things.
    // GPT-2 uses a standard deviation of 0.02 and we copy that.
    void fill_with_random_normal(float mean, float standard_deviation);

    // ────────────────────────────────
    // DEBUGGING HELP
    // Prints the matrix (or a corner of it) so you can eyeball values.
    // Nothing in the model calls this — it is purely for you while learning.
    // ────────────────────────────────
    void print(const std::string& label, int max_rows = 6, int max_cols = 8) const;
};


// ────────────────────────────────────────────────────────────────────────────────
// FREE FUNCTIONS THAT OPERATE ON MATRICES
//
// These are plain functions rather than member functions because "multiply A by
// B" is not really an action that belongs to A. Reading matrix_multiply(A, B)
// out loud tells you exactly what happens, which is the whole point.
//
// They all return a brand new Matrix by value. That copies memory, and yes, a
// production library would avoid it. C++17 moves the result rather than copying
// it in most cases, and the readability is worth far more to us than the cycles.
// ────────────────────────────────────────────────────────────────────────────────

// Standard matrix product.  A is [n, k], B is [k, m], result is [n, m].
// result(i, j) = sum over p of A(i, p) * B(p, j)
Matrix matrix_multiply(const Matrix& A, const Matrix& B);

// Flip rows and columns. A is [n, m], result is [m, n].
// We need this constantly in attention: K must be transposed before Q meets it.
Matrix matrix_transpose(const Matrix& A);

// Element-by-element addition. Both matrices must be exactly the same shape.
// This is how residual ("skip") connections are implemented.
Matrix matrix_add(const Matrix& A, const Matrix& B);

// Add one row vector to EVERY row of A.
// A is [n, m], row_vector is [1, m]. This is called BROADCASTING.
// It is how a bias gets added: one bias per column, shared by all tokens.
Matrix matrix_add_row_vector(const Matrix& A, const Matrix& row_vector);

// Multiply every element by a single number.
// Used for the 1/sqrt(head_dim) scaling inside attention.
Matrix matrix_scale(const Matrix& A, float scalar);

// Copy out a block of columns: columns [start_col, start_col + width).
// This is how we split the big [T, C] matrix into per-head [T, head_dim] slices.
Matrix matrix_slice_columns(const Matrix& A, int start_col, int width);

// The opposite of the above: write a small matrix into a block of columns of a
// bigger one, starting at start_col. This is how per-head results are
// concatenated back into a single [T, C] matrix.
void matrix_write_columns(Matrix& destination, const Matrix& source, int start_col);


// ────────────────────────────────────────────────────────────────────────────────
// SHARED RANDOM NUMBER ENGINE
//
// Every part of the program that needs randomness (weight init, batch sampling,
// text sampling) pulls from this one engine. One engine means one seed, which
// means a run can be reproduced exactly — priceless when you are debugging and
// need to know whether a change actually changed anything.
// ────────────────────────────────────────────────────────────────────────────────

// Returns a reference to the single, shared engine.
std::mt19937& get_random_engine();

// Set the seed so a run can be repeated exactly.
void set_random_seed(unsigned int seed);


#endif // __MATRIX_H__
