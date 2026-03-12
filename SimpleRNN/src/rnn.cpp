// This line tells the compiler: "Include the declarations from rnn.h"
// Without this, the compiler would not know what SimpleRNN is, what its member variables are,
// or what functions it has declared.
#include "rnn.h"


// ────────────────────────────────────────────────────────────────────────────────
// CONSTRUCTOR
// This is the function that gets called when you write: SimpleRNN my_rnn(1, 32);
// It creates a new RNN object and sets up its internal memory (weights & biases)
// ────────────────────────────────────────────────────────────────────────────────
SimpleRNN::SimpleRNN(int in_size, int hid_size)     // Parameters: input size & desired hidden layer size
    : input_size(in_size),                          // Store how many numbers come in each timestep (usually 1 = univariate)
      hidden_size(hid_size),                        // Store how many "memory cells" / neurons this RNN has in its hidden layer

      // Create the three main weight matrices with correct dimensions
      // Wxh = "input to hidden" weights – turns input vector into hidden vector contribution
      Wxh(hid_size, in_size),                       // Rows = hidden neurons, Columns = input features

      // Whh = "hidden to hidden" = recurrent weights – lets the RNN remember previous state
      Whh(hid_size, hidden_size),                   // Square matrix: hidden → hidden

      // Why = "hidden to output" weights – turns final hidden state into prediction
      // We use 1 output because we're predicting one number (next power value)
      Why(1, hid_size),                             // 1 row (one output), hidden_size columns

      // Bias vectors – small numbers added after matrix multiplication (helps model fit better)
      bh(hid_size),                                 // One bias per hidden neuron
      by(1)                                         // One bias for the single output
{
    // The body is empty because we used the initializer list above (more efficient)
    // No extra work needed after setting sizes
}


// ────────────────────────────────────────────────────────────────────────────────
// Initializes all weights and biases with small random numbers
// Very important: good initialization prevents the network from starting with crazy values
// ────────────────────────────────────────────────────────────────────────────────
void SimpleRNN::initialize_weights(double stddev)   // stddev = how spread out the random numbers should be (usually small like 0.1)
{
    // Create a source of true randomness (uses hardware if available)
    std::random_device rd;

    // Create a high-quality random number engine (Mersenne Twister)
    std::mt19937 gen(rd());

    // Create a normal (Gaussian) distribution: mean=0, standard deviation = stddev
    // Most values will be between -3*stddev and +3*stddev
    std::normal_distribution<double> dist(0.0, stddev);

    // ────────────────────────────────
    // Lambda function that fills ANY Eigen matrix with random numbers
    // This is like a mini-function we can reuse
    // ────────────────────────────────
    std::function<void(MatrixXd&)> fill_matrix = [&](MatrixXd& mat)
    {
        // Loop through every row
        for (int i = 0; i < mat.rows(); ++i)
        {
            // Loop through every column in that row
            for (int j = 0; j < mat.cols(); ++j)
            {
                // Assign a random number from our distribution
                mat(i, j) = dist(gen);
            }
        }
    };

    // Similar lambda, but for vectors (1D arrays)
    std::function<void(VectorXd&)> fill_vector = [&](VectorXd& vec)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            vec(i) = dist(gen);
        }
    };

    // Now actually fill every weight matrix and bias vector
    fill_matrix(Wxh);   // Input → hidden weights
    fill_matrix(Whh);   // Recurrent (memory) weights
    fill_matrix(Why);   // Hidden → prediction weights
    fill_vector(bh);    // Hidden layer bias
    fill_vector(by);    // Output bias
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD STEP – the heart of the RNN – computes one time step
// Takes current input x_t and previous memory h_prev
// Returns the new memory state h_t
// ────────────────────────────────────────────────────────────────────────────────
VectorXd SimpleRNN::forward_step(const VectorXd& x_t, const VectorXd& h_prev) const
{
    // Step 1: Compute linear combination
    // Wxh * x_t     → how much the current input contributes to hidden state
    // Whh * h_prev  → how much the previous memory contributes (this is the "recurrent" part)
    // + bh          → add bias to shift the result
    // Result so far: a vector of size hidden_size with raw scores

    // Step 2: Apply tanh activation function to every element
    // tanh squashes values to range [-1, 1]
    // This prevents numbers from growing too large or too small over many timesteps
    // .unaryExpr applies the lambda to every single element of the vector
    VectorXd hidden = (Wxh * x_t + Whh * h_prev + bh)
                       .unaryExpr([](double x)
                       {
                           return std::tanh(x);   // std::tanh is the hyperbolic tangent function
                       });

    // Return the new hidden state (memory) for this timestep
    return hidden;
}


// ────────────────────────────────────────────────────────────────────────────────
// FORWARD SEQUENCE – runs the RNN over many timesteps (whole sequence)
// Fills two output vectors by reference so caller can see all hidden states & predictions
// ────────────────────────────────────────────────────────────────────────────────
void SimpleRNN::forward_sequence(const std::vector<VectorXd>& inputs,
                                 std::vector<VectorXd>& hidden_states,
                                 std::vector<VectorXd>& outputs) const
{
    // Clear any old data that might be in these vectors
    hidden_states.clear();
    outputs.clear();

    // Start with zero memory (no previous information before first input)
    VectorXd h = VectorXd::Zero(hidden_size);  // all elements = 0.0

    // Loop through every input timestep
    for (const VectorXd& x : inputs)
    {
        // Compute next memory state using current input and previous memory
        h = forward_step(x, h);

        // Save this hidden state (useful later for training or visualization)
        hidden_states.push_back(h);

        // Compute prediction for this timestep
        // Why * h   → project hidden state to output space
        // + by      → add output bias
        VectorXd y_pred = Why * h + by;

        // Save the prediction
        outputs.push_back(y_pred);
    }
    // After the loop:
    // hidden_states has one entry per input timestep
    // outputs has one prediction per input timestep
}


// ────────────────────────────────────────────────────────────────────────────────
// LOSS FUNCTION – measures how wrong our predictions are
// Uses Mean Squared Error (MSE) – very common for regression problems
// ────────────────────────────────────────────────────────────────────────────────
double SimpleRNN::compute_loss(const std::vector<VectorXd>& targets,
                               const std::vector<VectorXd>& predictions) const
{
    // Safety check: we must have exactly the same number of targets and predictions
    if (targets.size() != predictions.size())
    {
        std::cout << "Target/prediction size mismatch\n";   // Print warning
        return 0.0;                                         // Return invalid value
    }

    double total_loss = 0.0;   // We'll add up all squared errors here

    // Loop over every timestep we have both target and prediction for
    for (size_t i = 0; i < targets.size(); ++i)
    {
        // Compute difference: target - prediction (vector subtraction)
        VectorXd diff = targets[i] - predictions[i];

        // squaredNorm() = sum of squares of all elements
        // Since our output is size 1, this is basically (target - pred)²
        total_loss += diff.squaredNorm();
    }

    // Return average squared error (Mean Squared Error)
    // This number is what we want to make SMALLER during training
    return total_loss / targets.size();
}