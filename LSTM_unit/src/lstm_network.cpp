#include <cstdio>
#include "lstm_network.h"




//*****************   Making A Prediction   *****************//

        // Feeds the input values of one training window through the
        // LSTM unit one at a time. The short term memory and long term
        // memory start at zero and are carried forward through the whole
        // window. The final short term memory is the prediction for the
        // NEXT data value.
        float Lstm_Network::compute_prediction(std::vector<float> &window, int window_size)
        {
            float short_term_memory = 0.0f;
            float long_term_memory = 0.0f;
            float new_short_term_memory = 0.0f;
            float new_long_term_memory = 0.0f;

            for (int i = 0; i < window_size; i++)
            {
                lstmUnit.compute_lstm_unit( window[i],
                                            short_term_memory,
                                            long_term_memory,
                                            new_short_term_memory,
                                            new_long_term_memory);

                short_term_memory = new_short_term_memory;
                long_term_memory = new_long_term_memory;
            }

            return short_term_memory;
        }


//*****************   Measuring The Error   *****************//

        // The last value in the window vector is the target - the data
        // value that actually came next in the file. The loss is the
        // squared difference between the prediction and the target.
        float Lstm_Network::compute_window_loss(std::vector<float> &window, int window_size)
        {
            float prediction = compute_prediction(window, window_size);
            float target = window[window_size];
            float error = target - prediction;

            return error * error;
        }


//*****************   Weight Access By Index   *****************//

        // The LSTM unit has 12 trainable values. Numbering them 0 to 11
        // lets the training loop treat them all the same way.
        float Lstm_Network::get_weight(int weight_index)
        {
            if (weight_index == 0) return lstmUnit.lstmUnitWeights.get_forget_gate_wxf();
            if (weight_index == 1) return lstmUnit.lstmUnitWeights.get_forget_gate_whf();
            if (weight_index == 2) return lstmUnit.lstmUnitWeights.get_forget_gate_bf();
            if (weight_index == 3) return lstmUnit.lstmUnitWeights.get_input_scaled_potential_memory_wxf();
            if (weight_index == 4) return lstmUnit.lstmUnitWeights.get_input_scaled_potential_memory_whf();
            if (weight_index == 5) return lstmUnit.lstmUnitWeights.get_input_scaled_potential_memory_bf();
            if (weight_index == 6) return lstmUnit.lstmUnitWeights.get_input_potential_memory_wxf();
            if (weight_index == 7) return lstmUnit.lstmUnitWeights.get_input_potential_memory_whf();
            if (weight_index == 8) return lstmUnit.lstmUnitWeights.get_input_potential_memory_bf();
            if (weight_index == 9) return lstmUnit.lstmUnitWeights.get_output_scaled_potential_memory_wxf();
            if (weight_index == 10) return lstmUnit.lstmUnitWeights.get_output_scaled_potential_memory_whf();
            return lstmUnit.lstmUnitWeights.get_output_scaled_potential_memory_bf();
        }

        int Lstm_Network::set_weight(int weight_index, float new_weight)
        {
            if (weight_index == 0) return lstmUnit.lstmUnitWeights.set_forget_gate_wxf(new_weight);
            if (weight_index == 1) return lstmUnit.lstmUnitWeights.set_forget_gate_whf(new_weight);
            if (weight_index == 2) return lstmUnit.lstmUnitWeights.set_forget_gate_bf(new_weight);
            if (weight_index == 3) return lstmUnit.lstmUnitWeights.set_input_scaled_potential_memory_wxf(new_weight);
            if (weight_index == 4) return lstmUnit.lstmUnitWeights.set_input_scaled_potential_memory_whf(new_weight);
            if (weight_index == 5) return lstmUnit.lstmUnitWeights.set_input_scaled_potential_memory_bf(new_weight);
            if (weight_index == 6) return lstmUnit.lstmUnitWeights.set_input_potential_memory_wxf(new_weight);
            if (weight_index == 7) return lstmUnit.lstmUnitWeights.set_input_potential_memory_whf(new_weight);
            if (weight_index == 8) return lstmUnit.lstmUnitWeights.set_input_potential_memory_bf(new_weight);
            if (weight_index == 9) return lstmUnit.lstmUnitWeights.set_output_scaled_potential_memory_wxf(new_weight);
            if (weight_index == 10) return lstmUnit.lstmUnitWeights.set_output_scaled_potential_memory_whf(new_weight);
            return lstmUnit.lstmUnitWeights.set_output_scaled_potential_memory_bf(new_weight);
        }


//*****************   Nudging One Weight   *****************//

        // Temporarily moves one weight by a small amount, measures the
        // loss with the weight in the new position, then puts the weight
        // back exactly where it was.
        float Lstm_Network::nudge_weight_and_get_loss(  std::vector<float> &window,
                                                        int window_size,
                                                        int weight_index,
                                                        float nudge_amount)
        {
            float original_weight = get_weight(weight_index);

            set_weight(weight_index, original_weight + nudge_amount);
            float loss_after_nudge = compute_window_loss(window, window_size);

            set_weight(weight_index, original_weight);

            return loss_after_nudge;
        }


//*****************   Training The Network   *****************//

        // Trains with the simplest form of gradient descent. For every
        // training window and every weight we ask one question: if this
        // weight moves up a tiny bit, does the loss go up or down? The
        // answer is the slope (the gradient):
        //
        //     slope = (loss_with_nudge - loss_now) / nudge_amount
        //
        // Then the weight takes a small step DOWNHILL:
        //
        //     new_weight = old_weight - learning_rate * slope
        //
        int Lstm_Network::train_network(    std::vector< std::vector<float> > &training_windows,
                                            int window_size,
                                            int number_of_epochs,
                                            float learning_rate)
        {
            float nudge_amount = 0.001f;
            int number_of_weights = 12;
            int number_of_windows = training_windows.size();

            for (int epoch = 0; epoch < number_of_epochs; epoch++)
            {
                float epoch_loss = 0.0f;

                for (int w_index = 0; w_index < number_of_windows; w_index++)
                {
                    float loss_now = compute_window_loss(training_windows[w_index], window_size);

                    // Measure the slope for every weight FIRST,
                    // then update all the weights together.
                    float slopes[12];

                    for (int w = 0; w < number_of_weights; w++)
                    {
                        float loss_nudged = nudge_weight_and_get_loss(  training_windows[w_index],
                                                                        window_size,
                                                                        w,
                                                                        nudge_amount);

                        slopes[w] = (loss_nudged - loss_now) / nudge_amount;
                    }

                    for (int w = 0; w < number_of_weights; w++)
                    {
                        float old_weight = get_weight(w);
                        set_weight(w, old_weight - learning_rate * slopes[w]);
                    }

                    epoch_loss = epoch_loss + loss_now;
                }

                printf("    Epoch %d   average loss = %f\n", epoch + 1, epoch_loss / number_of_windows);
            }

            return 0;
        }


//*****************   Scanning A Whole File   *****************//

        // Runs the LSTM continuously through the whole file, one data
        // point per step, never resetting the memories. After the unit
        // has seen data[i], its short term memory is the prediction for
        // data[i + 1]. The difference between that prediction and the
        // real data[i + 1] is stored in prediction_errors.
        int Lstm_Network::scan_file_errors( std::vector<float> &data,
                                            std::vector<float> &prediction_errors)
        {
            float short_term_memory = 0.0f;
            float long_term_memory = 0.0f;
            float new_short_term_memory = 0.0f;
            float new_long_term_memory = 0.0f;

            int number_of_values = data.size();

            prediction_errors.clear();

            for (int i = 0; i < number_of_values - 1; i++)
            {
                lstmUnit.compute_lstm_unit( data[i],
                                            short_term_memory,
                                            long_term_memory,
                                            new_short_term_memory,
                                            new_long_term_memory);

                short_term_memory = new_short_term_memory;
                long_term_memory = new_long_term_memory;

                // error belongs to data point i + 1
                prediction_errors.push_back(data[i + 1] - short_term_memory);
            }

            return 0;
        }
