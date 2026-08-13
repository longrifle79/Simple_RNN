#ifndef LSTM_NETWORK_H
#define LSTM_NETWORK_H

#include <vector>
#include "lstm_unit.h"


class Lstm_Network
{
    public:
        // A training window is a small vector holding WINDOW_SIZE
        // input values followed by 1 target value (the value that
        // came next in the file).
        float compute_prediction(std::vector<float> &window, int window_size);

        float compute_window_loss(std::vector<float> &window, int window_size);

        int train_network(  std::vector< std::vector<float> > &training_windows,
                            int window_size,
                            int number_of_epochs,
                            float learning_rate);

        // Runs the LSTM once through an entire file from start to end
        // (the memories are never reset) and fills prediction_errors
        // with (actual next value - predicted next value) at every point.
        int scan_file_errors(   std::vector<float> &data,
                                std::vector<float> &prediction_errors);

        Lstm lstmUnit;

    private:
        float nudge_weight_and_get_loss(    std::vector<float> &window,
                                            int window_size,
                                            int weight_index,
                                            float nudge_amount);

        float get_weight(int weight_index);
        int set_weight(int weight_index, float new_weight);
};

#endif
