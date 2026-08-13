/*
 * matrix.cpp — implementation of the small matrix class
 *
 * Nothing here is clever. That is intentional. Every loop is the loop you would
 * write on paper if someone asked you to multiply two matrices by hand.
 */

#include "matrix.h"

#include <iostream>     // std::cout for print()
#include <iomanip>      // std::setw / std::setprecision for tidy printing
#include <cassert>      // assert() — catches shape mistakes immediately


// ────────────────────────────────────────────────────────────────────────────────
// CONSTRUCTORS
// ────────────────────────────────────────────────────────────────────────────────

// Empty matrix. Exists so a class can declare "Matrix weights;" as a member and
// give it a real size later, inside its initialize() function.
Matrix::Matrix()
    : rows(0),      // no rows yet
      cols(0)       // no columns yet
{
    // values stays empty
}


// Sized matrix, zero filled.
Matrix::Matrix(int number_of_rows, int number_of_cols)
    : rows(number_of_rows),
      cols(number_of_cols),
      // Allocate rows*cols floats, all initialised to 0.0f by the vector
      // constructor's second argument.
      values(static_cast<size_t>(number_of_rows) * number_of_cols, 0.0f)
{
    // Body empty — the initialiser list above did all the work
}


// ────────────────────────────────────────────────────────────────────────────────
// ELEMENT ACCESS
// The single place where row-major index arithmetic happens. Everywhere else in
// the project just says at(r, c) and never thinks about layout again.
// ────────────────────────────────────────────────────────────────────────────────

float& Matrix::at(int row, int col)
{
    // Cheap safety net. Compile with -DNDEBUG to remove asserts from a fast build.
    assert(row >= 0 && row < rows);
    assert(col >= 0 && col < cols);

    // Row-major: skip past `row` complete rows, then step `col` across.
    return values[static_cast<size_t>(row) * cols + col];
}


const float& Matrix::at(int row, int col) const
{
    assert(row >= 0 && row < rows);
    assert(col >= 0 && col < cols);

    return values[static_cast<size_t>(row) * cols + col];
}


// ────────────────────────────────────────────────────────────────────────────────
// FILLING
// ────────────────────────────────────────────────────────────────────────────────

void Matrix::fill_with_zeros()
{
    // Walk the flat storage directly — no need for the 2D index here.
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = 0.0f;
    }
}


void Matrix::fill_with_ones()
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = 1.0f;
    }
}


// Fill with small random numbers from a Gaussian (bell curve) distribution.
//
// WHY RANDOM AT ALL: if every weight started at the same value, every neuron in
// a layer would compute the same thing and receive the same gradient forever.
// They would never differentiate. Random values break that symmetry.
//
// WHY SMALL: large initial weights push activations far from zero, where tanh
// and softmax saturate and gradients die. GPT-2 uses stddev = 0.02.
void Matrix::fill_with_random_normal(float mean, float standard_deviation)
{
    // Grab the one shared engine (see get_random_engine below).
    std::mt19937& engine = get_random_engine();

    // A normal distribution with the requested centre and spread.
    std::normal_distribution<float> distribution(mean, standard_deviation);

    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = distribution(engine);   // draw one sample per element
    }
}


// ────────────────────────────────────────────────────────────────────────────────
// PRINTING — a debugging convenience, never used by the model itself
// ────────────────────────────────────────────────────────────────────────────────

void Matrix::print(const std::string& label, int max_rows, int max_cols) const
{
    std::cout << label << "  [" << rows << " x " << cols << "]\n";

    // Only show a corner of large matrices — a [64 x 128] dump helps nobody.
    int rows_to_show = (rows < max_rows) ? rows : max_rows;
    int cols_to_show = (cols < max_cols) ? cols : max_cols;

    for (int r = 0; r < rows_to_show; ++r)
    {
        for (int c = 0; c < cols_to_show; ++c)
        {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << at(r, c);
        }

        // Tell the reader we truncated the row
        if (cols_to_show < cols)
        {
            std::cout << "   ...";
        }

        std::cout << "\n";
    }

    if (rows_to_show < rows)
    {
        std::cout << "   ...\n";
    }

    std::cout << "\n";
}


// ────────────────────────────────────────────────────────────────────────────────
// MATRIX MULTIPLY — the single most important function in the whole project
//
// A is [n, k],  B is [k, m],  result is [n, m]
//
//     result(i, j) = A(i, 0)*B(0, j) + A(i, 1)*B(1, j) + ... + A(i, k-1)*B(k-1, j)
//
// In words: element (i, j) of the answer is the dot product of row i of A with
// column j of B. The "inner" dimension k must match — that is the only rule.
//
// PERFORMANCE NOTE (deliberately not applied): swapping the j and p loops would
// walk memory in a more cache-friendly order and run several times faster. We
// keep the textbook ordering because "row of A times column of B" is exactly how
// you learned it, and clarity wins here. Build with -O2 or -O3 and it is fast
// enough for a model this size.
// ────────────────────────────────────────────────────────────────────────────────
Matrix matrix_multiply(const Matrix& A, const Matrix& B)
{
    // The inner dimensions must agree or the operation is meaningless.
    assert(A.cols == B.rows);

    // Result has A's rows and B's columns.
    Matrix result(A.rows, B.cols);

    for (int i = 0; i < A.rows; ++i)             // for each row of A
    {
        for (int j = 0; j < B.cols; ++j)         // for each column of B
        {
            float sum = 0.0f;                    // accumulate the dot product here

            for (int p = 0; p < A.cols; ++p)     // walk along the shared dimension
            {
                sum += A.at(i, p) * B.at(p, j);
            }

            result.at(i, j) = sum;
        }
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// TRANSPOSE — turn rows into columns
// A is [n, m] and the result is [m, n], with result(j, i) = A(i, j).
//
// Attention needs this: to compare every query against every key we compute
// Q * K^T, which is only shape-legal once K has been flipped.
// ────────────────────────────────────────────────────────────────────────────────
Matrix matrix_transpose(const Matrix& A)
{
    Matrix result(A.cols, A.rows);      // note the swapped dimensions

    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < A.cols; ++j)
        {
            result.at(j, i) = A.at(i, j);   // the swap itself
        }
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// ELEMENTWISE ADD — used for residual (skip) connections
// x + sublayer(x). Both operands must be exactly the same shape.
// ────────────────────────────────────────────────────────────────────────────────
Matrix matrix_add(const Matrix& A, const Matrix& B)
{
    assert(A.rows == B.rows);
    assert(A.cols == B.cols);

    Matrix result(A.rows, A.cols);

    // Same shape means we can work straight on the flat storage.
    for (size_t i = 0; i < A.values.size(); ++i)
    {
        result.values[i] = A.values[i] + B.values[i];
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// BROADCAST ADD — add one row vector to every row
//
// A is [n, m] and row_vector is [1, m]. Every row of A gets the same vector
// added to it. This is how a bias works: there is one bias number per output
// channel, and every token in the sequence receives the same one.
// ────────────────────────────────────────────────────────────────────────────────
Matrix matrix_add_row_vector(const Matrix& A, const Matrix& row_vector)
{
    assert(row_vector.rows == 1);          // it really must be a single row
    assert(row_vector.cols == A.cols);     // and match A's width

    Matrix result(A.rows, A.cols);

    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < A.cols; ++j)
        {
            // Same row_vector value reused for every row i — that is broadcasting
            result.at(i, j) = A.at(i, j) + row_vector.at(0, j);
        }
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// SCALE — multiply every element by one number
// Attention divides its scores by sqrt(head_dim) using this.
// ────────────────────────────────────────────────────────────────────────────────
Matrix matrix_scale(const Matrix& A, float scalar)
{
    Matrix result(A.rows, A.cols);

    for (size_t i = 0; i < A.values.size(); ++i)
    {
        result.values[i] = A.values[i] * scalar;
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// SLICE COLUMNS — pull out a vertical strip of a matrix
//
// Multi-head attention takes a [T, C] matrix and treats it as num_heads separate
// [T, head_dim] matrices sitting side by side. Head h owns the columns
// [h*head_dim, (h+1)*head_dim). This function hands you head h's strip.
//
// Real implementations reshape a tensor instead of copying. Copying is slower
// but you can actually see what a "head" is, which is the point.
// ────────────────────────────────────────────────────────────────────────────────
Matrix matrix_slice_columns(const Matrix& A, int start_col, int width)
{
    assert(start_col >= 0);
    assert(start_col + width <= A.cols);

    Matrix result(A.rows, width);

    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // Column j of the slice is column (start_col + j) of the original
            result.at(i, j) = A.at(i, start_col + j);
        }
    }

    return result;
}


// ────────────────────────────────────────────────────────────────────────────────
// WRITE COLUMNS — the inverse of the slice above
//
// After each head has produced its own [T, head_dim] output, we glue them back
// together side by side into one [T, C] matrix. This function writes one head's
// result into its slot.
// ────────────────────────────────────────────────────────────────────────────────
void matrix_write_columns(Matrix& destination, const Matrix& source, int start_col)
{
    assert(destination.rows == source.rows);
    assert(start_col + source.cols <= destination.cols);

    for (int i = 0; i < source.rows; ++i)
    {
        for (int j = 0; j < source.cols; ++j)
        {
            destination.at(i, start_col + j) = source.at(i, j);
        }
    }
}


// ────────────────────────────────────────────────────────────────────────────────
// THE SHARED RANDOM ENGINE
//
// A function-local static is created once, the first time the function is
// called, and then lives for the rest of the program. That gives us exactly one
// engine without needing a global variable floating around.
//
// Default seed is fixed (1234) rather than random, so two runs of the program
// behave identically. When you are hunting a bug you want the same numbers every
// time. Call set_random_seed(std::random_device{}()) if you ever want real
// randomness.
// ────────────────────────────────────────────────────────────────────────────────
std::mt19937& get_random_engine()
{
    static std::mt19937 engine(1234);   // created once, on first use
    return engine;
}


void set_random_seed(unsigned int seed)
{
    get_random_engine().seed(seed);
}
