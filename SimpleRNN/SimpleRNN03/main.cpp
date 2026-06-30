/*
 * Simple RNN — anomaly classifier for time-series CSV data
 *
 * Same coding style as the MNIST example: a flat `Mat` (vector + dims) and a
 * handful of small free functions. The only new idea versus MNIST is the time
 * loop — a vanilla (Elman) recurrent layer reads the signal one value at a
 * time, carrying a hidden state forward, then we classify the FINAL hidden
 * state as Normal or Anomaly. So it is literally the MNIST network with a
 * loop wrapped around the hidden layer.
 *
 * Architecture:   1 value/step  ->  HIDDEN (tanh, recurrent)  ->  2 (softmax)
 * Training:       Full-batch gradient descent + backprop-through-time (BPTT)
 *
 * Data layout (one CSV = one labelled example, same as MNIST one image = one
 * sample):
 *     .../data/Normal/equation_data_00001.csv    -> label 0  (normal)
 *     .../data/Anamoly/equation_data_00001.csv   -> label 1  (anomaly)
 *
 * Each CSV is a "Power,Time" file; we read the Power column and resample it to
 * a fixed length SEQ_LEN so every sequence is the same size (like every MNIST
 * image being 784 pixels).
 *
 * Build:
 *   g++ -O2 -std=c++17 -o simple_rnn main.cpp
 *   (older compilers may also need:  -lstdc++fs )
 *
 * Run:
 *   ./simple_rnn                       (uses the two paths set below)
 *   ./simple_rnn <normal_dir> <anomaly_dir>   (override the paths)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;


#define NUM_OF_ITERATIONS 400
// ─── Where the data lives (edit these, or pass them on the command line) ──────
string NORMAL_DIR  = "/home/gary/Desktop/Simple_RNN/SimpleRNN/SimpleRNN03/data/Normal_2";
string ANOMALY_DIR = "/home/gary/Desktop/Simple_RNN/SimpleRNN/SimpleRNN03/data/Anamoly_2";

// ─── Constants ────────────────────────────────────────────────────────────────
const int SEQ_LEN = 60;    // every sequence is resampled to this many steps
const int HIDDEN  = 32;    // hidden units in the recurrent layer
const int OUTPUT  = 2;     // 0 = normal, 1 = anomaly
const int DEV     = 200;   // how many samples to hold out for validation

// ─── Simple matrix: just a flat vector + dimensions (same as MNIST) ───────────
struct Mat
{
    int rows, cols;
    vector<float> d; // row-major: element [r][c] = d[r*cols + c]

    Mat() : rows(0), cols(0) {}
    Mat(int r, int c, float fill = 0.f) : rows(r), cols(c), d(r * c, fill) {}

    float& at(int r, int c)       { return d[r * cols + c]; }
    float  at(int r, int c) const { return d[r * cols + c]; }
};

// ─── Matrix multiply: C = A * B ───────────────────────────────────────────────
Mat matmul(const Mat& A, const Mat& B)
{
    assert(A.cols == B.rows);
    Mat C(A.rows, B.cols, 0.f);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j)
                C.at(i, j) += A.at(i, k) * B.at(k, j);
    return C;
}

// ─── Matrix transpose ─────────────────────────────────────────────────────────
Mat T(const Mat& A)
{
    Mat B(A.cols, A.rows);
    for (int r = 0; r < A.rows; ++r)
        for (int c = 0; c < A.cols; ++c)
            B.at(c, r) = A.at(r, c);
    return B;
}

// ─── Element-wise multiply ────────────────────────────────────────────────────
Mat elemul(const Mat& A, const Mat& B)
{
    assert(A.rows == B.rows && A.cols == B.cols);
    Mat C(A.rows, A.cols);
    for (int i = 0; i < (int)A.d.size(); ++i)
        C.d[i] = A.d[i] * B.d[i];
    return C;
}

// ─── Scale every element by a scalar ─────────────────────────────────────────
Mat scale(const Mat& A, float s)
{
    Mat C = A;
    for (float& v : C.d) v *= s;
    return C;
}

// ─── Add: A + B ──────────────────────────────────────────────────────────────
Mat add(const Mat& A, const Mat& B)
{
    assert(A.rows == B.rows && A.cols == B.cols);
    Mat C(A.rows, A.cols);
    for (int i = 0; i < (int)A.d.size(); ++i)
        C.d[i] = A.d[i] + B.d[i];
    return C;
}

// ─── Subtract: A - B ─────────────────────────────────────────────────────────
Mat sub(const Mat& A, const Mat& B)
{
    assert(A.rows == B.rows && A.cols == B.cols);
    Mat C(A.rows, A.cols);
    for (int i = 0; i < (int)A.d.size(); ++i)
        C.d[i] = A.d[i] - B.d[i];
    return C;
}

// ─── Add bias: A (rows x cols) + b (rows x 1), broadcast over columns ─────────
Mat add_bias(const Mat& A, const Mat& b)
{
    assert(A.rows == b.rows && b.cols == 1);
    Mat C = A;
    for (int r = 0; r < A.rows; ++r)
        for (int c = 0; c < A.cols; ++c)
            C.at(r, c) += b.at(r, 0);
    return C;
}

// ─── Sum each row into a (rows x 1) column vector ─────────────────────────────
Mat row_sum(const Mat& A)
{
    Mat S(A.rows, 1, 0.f);
    for (int r = 0; r < A.rows; ++r)
        for (int c = 0; c < A.cols; ++c)
            S.at(r, 0) += A.at(r, c);
    return S;
}

// ─── tanh applied element-wise (the recurrent activation) ─────────────────────
Mat tanh_mat(const Mat& Z)
{
    Mat A = Z;
    for (float& v : A.d) v = tanhf(v);
    return A;
}

// ─── tanh derivative, given the tanh OUTPUT h:  1 - h^2  (element-wise) ────────
Mat dtanh(const Mat& H)
{
    Mat D(H.rows, H.cols);
    for (int i = 0; i < (int)H.d.size(); ++i)
        D.d[i] = 1.f - H.d[i] * H.d[i];
    return D;
}

// ─── Softmax: applied column-by-column (each column = one sample) ─────────────
Mat softmax(const Mat& Z)
{
    Mat A(Z.rows, Z.cols);
    for (int c = 0; c < Z.cols; ++c)
    {
        float mx = Z.at(0, c);
        for (int r = 1; r < Z.rows; ++r) mx = max(mx, Z.at(r, c));

        float sum = 0.f;
        for (int r = 0; r < Z.rows; ++r)
        {
            A.at(r, c) = expf(Z.at(r, c) - mx);
            sum += A.at(r, c);
        }
        for (int r = 0; r < Z.rows; ++r) A.at(r, c) /= sum;
    }
    return A;
}

// ─── One-hot encode labels → Mat(OUTPUT, m) ───────────────────────────────────
Mat one_hot(const vector<int>& Y)
{
    Mat OH(OUTPUT, (int)Y.size(), 0.f);
    for (int c = 0; c < (int)Y.size(); ++c)
        OH.at(Y[c], c) = 1.f;
    return OH;
}

// ─── Accuracy: fraction of argmax(A2[:,i]) == Y[i] ───────────────────────────
float accuracy(const Mat& A2, const vector<int>& Y)
{
    int correct = 0;
    for (int c = 0; c < A2.cols; ++c)
    {
        int pred = 0;
        for (int r = 1; r < A2.rows; ++r)
            if (A2.at(r, c) > A2.at(pred, c)) pred = r;
        if (pred == Y[c]) ++correct;
    }
    return (float)correct / A2.cols;
}

// ─── Random initialisation: values in [-0.5, 0.5] (same as MNIST) ────────────
Mat rand_mat(int rows, int cols, mt19937& rng)
{
    uniform_real_distribution<float> dist(-0.5f, 0.5f);
    Mat M(rows, cols);
    for (float& v : M.d) v = dist(rng);
    return M;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Data loading
// ═════════════════════════════════════════════════════════════════════════════

// ─── List every .csv in a directory, sorted by name ──────────────────────────
vector<string> list_csv(const string& dir)
{
    vector<string> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".csv")
            files.push_back(e.path().string());
    sort(files.begin(), files.end());
    return files;
}

// ─── Read the Power column from one "Power,Time" CSV ──────────────────────────
vector<float> load_power(const string& path)
{
    ifstream f(path);
    assert(f && "Cannot open CSV file");

    vector<float> power;
    string line;
    bool first = true;
    while (getline(f, line))
    {
        if (line.empty()) continue;
        // grab the text before the first comma (the Power column)
        string tok = line.substr(0, line.find(','));
        try { power.push_back(stof(tok)); }
        catch (...) { if (first) { /* header row */ } }   // skip non-numeric header
        first = false;
    }
    return power;
}

// ─── Resample a sequence to exactly L points (linear interpolation) ───────────
//  This makes every file the same length, the way every MNIST image is 784 px.
vector<float> resample(const vector<float>& v, int L)
{
    vector<float> out(L, 0.f);
    if (v.empty()) return out;
    if ((int)v.size() == 1) { fill(out.begin(), out.end(), v[0]); return out; }

    for (int i = 0; i < L; ++i)
    {
        float pos = (float)i * (v.size() - 1) / (L - 1); // map i -> [0, N-1]
        int   a   = (int)pos;
        int   b   = min(a + 1, (int)v.size() - 1);
        float frac = pos - a;
        out[i] = v[a] * (1.f - frac) + v[b] * frac;
    }
    return out;
}

// ─── One labelled example: its (resampled) sequence and its class ─────────────
struct Sample
{
    vector<float> seq;   // length SEQ_LEN
    int           label; // 0 normal, 1 anomaly
};

// ─── Load every CSV in a directory, resample, and tag with `label` ────────────
void load_dir(const string& dir, int label, vector<Sample>& out)
{
    vector<string> files = list_csv(dir);
    cout << "  " << dir << "  ->  " << files.size() << " files\n";
    for (const string& path : files)
        out.push_back({ resample(load_power(path), SEQ_LEN), label });
}

// ═════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    if (argc >= 3) { NORMAL_DIR = argv[1]; ANOMALY_DIR = argv[2]; }

    // ── 1. Load every file from both folders ─────────────────────────────────
    cout << "Loading data...\n";
    vector<Sample> data;
    load_dir(NORMAL_DIR,  0, data);
    load_dir(ANOMALY_DIR, 1, data);
    assert(!data.empty() && "No data found - check the directory paths");

    // ── 2. Normalise the Power values (mean 0, std 1), like MNIST's /255 ─────
    //  We compute one global mean/std over all data and reuse it everywhere.
    double sum = 0, sumsq = 0; long n = 0;
    for (const auto& s : data) for (float v : s.seq) { sum += v; sumsq += (double)v*v; ++n; }
    float mu  = (float)(sum / n);
    float sig = (float)sqrt(sumsq / n - (double)mu * mu);
    if (sig < 1e-6f) sig = 1.f;
    for (auto& s : data) for (float& v : s.seq) v = (v - mu) / sig;
    cout << "Normalised Power with mean=" << mu << " std=" << sig << "\n";

    // ── 3. Shuffle, then split into train / dev (same idea as MNIST) ─────────
    mt19937 rng(42);
    shuffle(data.begin(), data.end(), rng);

    int dev_n   = min(DEV, (int)data.size() / 5);
    int train_n = (int)data.size() - dev_n;

    // Pack sequences into matrices. X is (SEQ_LEN x m): row t holds the value
    // at timestep t for every sample, so one "column" is one whole sequence.
    auto pack = [&](int begin, int count, Mat& X, vector<int>& Y)
    {
        X = Mat(SEQ_LEN, count);
        Y.resize(count);
        for (int c = 0; c < count; ++c)
        {
            const Sample& s = data[begin + c];
            Y[c] = s.label;
            for (int t = 0; t < SEQ_LEN; ++t) X.at(t, c) = s.seq[t];
        }
    };
    Mat X_train, X_dev; vector<int> Y_train, Y_dev;
    pack(0,        train_n, X_train, Y_train);
    pack(train_n,  dev_n,   X_dev,   Y_dev);

    cout << "Training samples: " << train_n << "\n";
    cout << "Dev samples:      " << dev_n   << "\n\n";

    // ── 4. Initialise weights and biases ─────────────────────────────────────
    //  Wx (HIDDEN x 1)      input  -> hidden
    //  Wh (HIDDEN x HIDDEN) hidden -> hidden  (the recurrent weights)
    //  bh (HIDDEN x 1)
    //  Wy (OUTPUT x HIDDEN) hidden -> output
    //  by (OUTPUT x 1)
    Mat Wx = rand_mat(HIDDEN, 1,      rng);
    Mat Wh = scale(rand_mat(HIDDEN, HIDDEN, rng), 0.4f); // spectral radius < 1 -> stable
    Mat bh = Mat(HIDDEN, 1, 0.f);                        // biases start at zero
    Mat Wy = scale(rand_mat(OUTPUT, HIDDEN, rng), 0.2f); // small -> balanced start
    Mat by = Mat(OUTPUT, 1, 0.f);

    // ── 5. Training loop ──────────────────────────────────────────────────────
    const int   ITERATIONS = NUM_OF_ITERATIONS; // how many times to loop through the full batch
    const float ALPHA      = 0.03f;          // learning rate
    const float INV_M      = 1.f / train_n;
    Mat OH = one_hot(Y_train);               // targets never change

    for (int iter = 0; iter <= ITERATIONS; ++iter)
    {
        // ── Forward propagation through time ─────────────────────────────────
        //  h_0 = 0
        //  for t = 0..SEQ_LEN-1:
        //      h_t = tanh( Wx * x_t  +  Wh * h_{t-1}  +  bh )
        //  h_max = element-wise MAX of all h_t over time  ← "most unusual moment"
        //  Z     = Wy * h_max + by
        //  A2    = softmax(Z)                              ← class probabilities
        //
        //  Max-pooling (not averaging) is the key choice for anomaly detection:
        //  a brief unusual moment survives the pool instead of being averaged
        //  away, and on the backward pass the gradient goes straight to the
        //  timestep that produced the peak. We remember that timestep in argmax.
        vector<Mat> H(SEQ_LEN);
        Mat h_prev(HIDDEN, train_n, 0.f);            // h_0 = zeros
        Mat h_max(HIDDEN, train_n, -1e30f);          // running max over time
        vector<int> argmax(HIDDEN * train_n, 0);     // which t won, per element
        for (int t = 0; t < SEQ_LEN; ++t)
        {
            // x_t : the t-th value of every sequence, shaped (1 x m)
            Mat x_t(1, train_n);
            for (int c = 0; c < train_n; ++c) x_t.at(0, c) = X_train.at(t, c);

            Mat a = add_bias(add(matmul(Wx, x_t), matmul(Wh, h_prev)), bh);
            H[t]   = tanh_mat(a);
            for (int i = 0; i < (int)H[t].d.size(); ++i)
                if (H[t].d[i] > h_max.d[i]) { h_max.d[i] = H[t].d[i]; argmax[i] = t; }
            h_prev = H[t];
        }

        Mat Z2 = add_bias(matmul(Wy, h_max), by);
        Mat A2 = softmax(Z2);

        // ── Print loss + accuracy every 20 iterations ───────────────────────
        if (iter % 20 == 0)
        {
            float L = 0.f;
            for (int c = 0; c < train_n; ++c)
                L -= logf(max(A2.at(Y_train[c], c), 1e-7f));
            cout << "Iteration " << iter
                 << "  loss: " << L / train_n
                 << "  accuracy: " << accuracy(A2, Y_train) * 100.f << "%\n";
        }

        // ── Backward propagation through time ────────────────────────────────
        //  dZ2    = A2 - one_hot(Y)
        //  dWy    = (1/m) dZ2 · h_maxᵀ ,  dby = (1/m) Σ dZ2
        //  dh_max = Wyᵀ · dZ2                       ← gradient into the max
        //  Each element of dh_max belongs to exactly ONE timestep — the one
        //  that produced the max (argmax). So as we walk backwards, a step's
        //  direct gradient is dh_max where argmax == t, and 0 elsewhere, PLUS
        //  the recurrent gradient coming back from t+1.
        //      da   = (direct + recurrent) ⊙ (1 - h_t²)   (tanh derivative)
        //      dWx += da · x_tᵀ ,  dWh += da · h_{t-1}ᵀ ,  dbh += Σ da
        //      recurrent = Whᵀ · da                 ← gradient into h_{t-1}
        Mat dZ2 = sub(A2, OH);
        Mat dWy = scale(matmul(dZ2, T(h_max)), INV_M);
        Mat dby = scale(row_sum(dZ2),          INV_M);

        Mat dh_max = matmul(T(Wy), dZ2);              // gradient into h_max

        Mat recurrent(HIDDEN, train_n, 0.f);          // gradient from the future
        Mat dWx(HIDDEN, 1, 0.f), dWh(HIDDEN, HIDDEN, 0.f), dbh(HIDDEN, 1, 0.f);

        for (int t = SEQ_LEN - 1; t >= 0; --t)
        {
            // direct: route dh_max only to the timestep that produced the max
            Mat direct(HIDDEN, train_n, 0.f);
            for (int i = 0; i < (int)direct.d.size(); ++i)
                if (argmax[i] == t) direct.d[i] = dh_max.d[i];

            Mat dh = add(direct, recurrent);          // direct + recurrent
            Mat da = elemul(dh, dtanh(H[t]));         // through the tanh

            Mat x_t(1, train_n);
            for (int c = 0; c < train_n; ++c) x_t.at(0, c) = X_train.at(t, c);
            Mat h_prev = (t == 0) ? Mat(HIDDEN, train_n, 0.f) : H[t - 1];

            dWx = add(dWx, matmul(da, T(x_t)));
            dWh = add(dWh, matmul(da, T(h_prev)));
            dbh = add(dbh, row_sum(da));

            recurrent = matmul(T(Wh), da);            // pass gradient back a step
        }
        dWx = scale(dWx, INV_M);
        dWh = scale(dWh, INV_M);
        dbh = scale(dbh, INV_M);

        // ── Parameter update: W := W - alpha * dW ────────────────────────────
        for (int i = 0; i < (int)Wx.d.size(); ++i) Wx.d[i] -= ALPHA * dWx.d[i];
        for (int i = 0; i < (int)Wh.d.size(); ++i) Wh.d[i] -= ALPHA * dWh.d[i];
        for (int i = 0; i < (int)bh.d.size(); ++i) bh.d[i] -= ALPHA * dbh.d[i];
        for (int i = 0; i < (int)Wy.d.size(); ++i) Wy.d[i] -= ALPHA * dWy.d[i];
        for (int i = 0; i < (int)by.d.size(); ++i) by.d[i] -= ALPHA * dby.d[i];
    }

    // ── 6. Evaluate on the dev set ────────────────────────────────────────────
    Mat h_prev(HIDDEN, dev_n, 0.f);
    Mat h_max(HIDDEN, dev_n, -1e30f);
    for (int t = 0; t < SEQ_LEN; ++t)
    {
        Mat x_t(1, dev_n);
        for (int c = 0; c < dev_n; ++c) x_t.at(0, c) = X_dev.at(t, c);
        h_prev = tanh_mat(add_bias(add(matmul(Wx, x_t), matmul(Wh, h_prev)), bh));
        for (int i = 0; i < (int)h_prev.d.size(); ++i)
            if (h_prev.d[i] > h_max.d[i]) h_max.d[i] = h_prev.d[i];
    }
    Mat A2_dev = softmax(add_bias(matmul(Wy, h_max), by));
    cout << "\nDev set accuracy: " << accuracy(A2_dev, Y_dev) * 100.f << "%\n";

    return 0;
}