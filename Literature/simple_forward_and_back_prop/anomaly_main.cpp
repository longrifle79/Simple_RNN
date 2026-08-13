#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "lstm_network.h"


// The plan:
//
//   1. Read the Power values from the csv file.
//   2. Scale them down to the range 0 to 1 (the LSTM works
//      best with small numbers).
//   3. Train the LSTM to predict the NEXT value from the
//      previous WINDOW_SIZE values, using a clean stretch of
//      the data for training.
//   4. Slide over the whole file making predictions. Where the
//      data is normal the prediction error is small (just the
//      noise). Where something abnormal happens the prediction
//      error jumps.
//   5. Flag every point whose error is far beyond the normal
//      noise level, and group neighboring flagged points into
//      anomaly regions.


//*****************   Settings   *****************//

        const int WINDOW_SIZE = 20;             // values the LSTM reads before predicting
        const int TRAIN_START = 0;              // train on the first part of the file
        const int TRAIN_END = 120000;           // covers one full cycle of the wave
        const int NUMBER_OF_EPOCHS = 6;
        const float LEARNING_RATE = 0.05f;
        const float SCALE = 100.0f;             // Power is 0..100, so divide by 100
        const float THRESHOLD_SIGMAS = 6.0f;    // how far past normal noise = anomaly
        const int GAP_TOLERANCE = 1500;         // flagged points closer than this merge into one region


//*****************   Reading The Data File   *****************//

        int read_csv_file(  const char *file_name,
                            std::vector<float> &power_values,
                            std::vector<float> &time_values)
        {
            std::ifstream data_file(file_name);

            if (!data_file.is_open())
            {
                printf("Could not open file: %s\n", file_name);
                return 1;
            }

            std::string line;

            // Skip the header line (Power,Time)
            std::getline(data_file, line);

            while (std::getline(data_file, line))
            {
                if (line.size() == 0)
                {
                    continue;
                }

                std::stringstream line_stream(line);
                std::string power_text;
                std::string time_text;

                std::getline(line_stream, power_text, ',');
                std::getline(line_stream, time_text, ',');

                power_values.push_back(atof(power_text.c_str()));
                time_values.push_back(atof(time_text.c_str()));
            }

            data_file.close();
            return 0;
        }


//*****************   Main Program   *****************//

        int main(int argc, char *argv[])
        {
            const char *file_name = "equation_data_00001.csv";

            if (argc > 1)
            {
                file_name = argv[1];
            }

            srand(42);      // same random starting weights every run


            // ─── Step 1: read the data ───
            std::vector<float> power_values;
            std::vector<float> time_values;

            if (read_csv_file(file_name, power_values, time_values) != 0)
            {
                return 1;
            }

            int number_of_values = power_values.size();
            printf("Read %d data points from %s\n\n", number_of_values, file_name);


            // ─── Step 2: scale the data down to 0..1 ───
            std::vector<float> scaled_values;

            for (int i = 0; i < number_of_values; i++)
            {
                scaled_values.push_back(power_values[i] / SCALE);
            }


            // ─── Step 3: train the LSTM on a clean stretch ───
            printf("Training the LSTM on points %d to %d ...\n", TRAIN_START, TRAIN_END);

            Lstm_Network network;

            network.train_network(  scaled_values,
                                    TRAIN_START,
                                    TRAIN_END,
                                    WINDOW_SIZE,
                                    NUMBER_OF_EPOCHS,
                                    LEARNING_RATE);

            printf("Training done.\n\n");


            // ─── Step 4: measure the prediction error at every point ───
            printf("Scanning the whole file for anomalies ...\n");

            std::vector<float> prediction_errors;

            for (int i = 0; i < number_of_values - WINDOW_SIZE; i++)
            {
                float prediction = network.compute_prediction(scaled_values, i, WINDOW_SIZE);
                float actual = scaled_values[i + WINDOW_SIZE];
                prediction_errors.push_back(actual - prediction);
            }

            int number_of_errors = prediction_errors.size();


            // ─── Step 5: compare each error to its NEIGHBORS ───
            //
            // Even a well trained network has a little slow-changing
            // error that drifts with the wave. That slow drift is not an
            // anomaly. So instead of comparing each error to one global
            // average, we compare it to the average of its own
            // neighborhood. Slow drift cancels out; sudden jumps stand out.

            int neighborhood = 150;     // look this many points to each side

            std::vector<float> local_deviations;

            // running_sum holds the sum of the errors inside the
            // neighborhood window as it slides along
            float running_sum = 0.0f;
            int window_left = 0;
            int window_right = 0;       // one past the last point in the window

            for (int i = 0; i < number_of_errors; i++)
            {
                int want_left = i - neighborhood;
                int want_right = i + neighborhood + 1;

                if (want_left < 0)
                {
                    want_left = 0;
                }

                if (want_right > number_of_errors)
                {
                    want_right = number_of_errors;
                }

                // grow the window on the right
                while (window_right < want_right)
                {
                    running_sum = running_sum + prediction_errors[window_right];
                    window_right = window_right + 1;
                }

                // shrink the window on the left
                while (window_left < want_left)
                {
                    running_sum = running_sum - prediction_errors[window_left];
                    window_left = window_left + 1;
                }

                float local_mean = running_sum / (window_right - window_left);
                local_deviations.push_back(prediction_errors[i] - local_mean);
            }

            // Normal noise level of the local deviations
            float squared_sum = 0.0f;

            for (int i = 0; i < number_of_errors; i++)
            {
                squared_sum = squared_sum + local_deviations[i] * local_deviations[i];
            }

            float deviation_std = sqrt(squared_sum / number_of_errors);

            printf("Local deviation standard deviation = %f\n", deviation_std);

            float threshold = THRESHOLD_SIGMAS * deviation_std;

            printf("Anomaly threshold = %f (%.1f standard deviations)\n\n", threshold, THRESHOLD_SIGMAS);


            // ─── Step 6: flag the points beyond the threshold ───
            std::vector<int> flagged_points;

            for (int i = 0; i < number_of_errors; i++)
            {
                float deviation = local_deviations[i];

                if (deviation < 0.0f)
                {
                    deviation = -deviation;
                }

                if (deviation > threshold)
                {
                    // the error at position i belongs to data point i + WINDOW_SIZE
                    flagged_points.push_back(i + WINDOW_SIZE);
                }
            }

            printf("Flagged %d individual points.\n\n", (int)flagged_points.size());


            // ─── Step 7: group neighboring flagged points into regions ───
            if (flagged_points.size() == 0)
            {
                printf("No anomalies found.\n");
                return 0;
            }

            printf("─────────────  ANOMALY REPORT  ─────────────\n");

            int region_start = flagged_points[0];
            int region_end = flagged_points[0];
            int region_number = 1;

            for (int i = 1; i <= (int)flagged_points.size(); i++)
            {
                int is_last_point = (i == (int)flagged_points.size());

                if (!is_last_point && flagged_points[i] - region_end <= GAP_TOLERANCE)
                {
                    // still inside the same region
                    region_end = flagged_points[i];
                }
                else
                {
                    // the region just ended - report it
                    printf("Anomaly %d:  data points %d to %d   (time %.6f to %.6f seconds)\n",
                            region_number,
                            region_start,
                            region_end,
                            time_values[region_start],
                            time_values[region_end]);

                    region_number = region_number + 1;

                    if (!is_last_point)
                    {
                        region_start = flagged_points[i];
                        region_end = flagged_points[i];
                    }
                }
            }

            printf("─────────────────────────────────────────────\n");

            return 0;
        }
