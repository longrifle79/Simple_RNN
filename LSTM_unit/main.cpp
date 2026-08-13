#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include "lstm_network.h"


// The plan:
//
//   1. List the csv files in the Normal directory and the
//      Anomaly directory.
//   2. Cut training windows out of the Normal files. Only the
//      Normal files are used for training - this teaches the
//      LSTM what NORMAL data looks like.
//   3. Train the LSTM to predict the next value from the
//      previous WINDOW_SIZE values.
//   4. Scan every file in BOTH directories. For each file the
//      LSTM runs through the whole file predicting the next
//      value at every point. Where the file behaves normally
//      the prediction error is just noise. Where an anomaly
//      happens the error jumps.
//   5. Give every file one anomaly score: its largest error
//      jump, measured in standard deviations. Files scoring
//      past the threshold are declared ANOMALY, the rest are
//      declared NORMAL. The location of the largest jump is
//      reported for every anomaly.


//*****************   Settings   *****************//

        const int WINDOW_SIZE = 20;             // values the LSTM reads before predicting
        const int WINDOW_SPACING = 2000;        // distance between training windows in a file
        const int NUMBER_OF_EPOCHS = 100;
        const float LEARNING_RATE = 0.05f;
        const float SCALE = 100.0f;             // Power is 0..100, so divide by 100
        const int NEIGHBORHOOD = 150;           // errors are compared to this many neighbors each side
        const int WARM_UP = 200;                // skip the first errors while the memories settle
        const float THRESHOLD_SIGMAS = 6.0f;    // score past this = anomaly

        const char *NORMAL_DIRECTORY = "/home/gary/Desktop/Simple_RNN/SimpleRNN/SimpleRNN03/data/Normal_3";
        const char *ANOMALY_DIRECTORY = "/home/gary/Desktop/Simple_RNN/SimpleRNN/SimpleRNN03/data/Anamoly_3";


//*****************   Listing The CSV Files In A Directory   *****************//

        int list_csv_files( const char *directory_path,
                            std::vector<std::string> &file_paths)
        {
            DIR *directory = opendir(directory_path);

            if (directory == NULL)
            {
                printf("Could not open directory: %s\n", directory_path);
                return 1;
            }

            struct dirent *entry;

            while ((entry = readdir(directory)) != NULL)
            {
                std::string file_name = entry->d_name;

                // keep only names ending in .csv
                if (file_name.size() > 4 &&
                    file_name.compare(file_name.size() - 4, 4, ".csv") == 0)
                {
                    std::string full_path = std::string(directory_path) + "/" + file_name;
                    file_paths.push_back(full_path);
                }
            }

            closedir(directory);

            // sort the paths so every run processes the files in the same order
            std::sort(file_paths.begin(), file_paths.end());

            return 0;
        }


//*****************   Reading One Data File   *****************//

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

                power_values.push_back(atof(power_text.c_str()) / SCALE);
                time_values.push_back(atof(time_text.c_str()));
            }

            data_file.close();
            return 0;
        }


//*****************   Cutting Training Windows From A File   *****************//

        // Slides along the file taking a window of WINDOW_SIZE inputs
        // plus 1 target every WINDOW_SPACING points. Each window becomes
        // one training example.
        int cut_training_windows(   std::vector<float> &power_values,
                                    std::vector< std::vector<float> > &training_windows)
        {
            int number_of_values = power_values.size();

            for (int start = 0;
                 start + WINDOW_SIZE + 1 < number_of_values;
                 start = start + WINDOW_SPACING)
            {
                std::vector<float> window;

                for (int i = 0; i < WINDOW_SIZE + 1; i++)
                {
                    window.push_back(power_values[start + i]);
                }

                training_windows.push_back(window);
            }

            return 0;
        }


//*****************   Scoring One File   *****************//

        // Turns a file's prediction errors into one anomaly score.
        //
        // Each error is compared to the average of its own neighborhood,
        // so slow drift cancels out and sudden jumps stand out. The score
        // is the largest jump measured in standard deviations. The
        // location of that largest jump is also returned.
        float score_file(   std::vector<float> &prediction_errors,
                            int &worst_point)
        {
            int number_of_errors = prediction_errors.size();

            // local deviation = error minus the average of its neighbors
            std::vector<float> local_deviations;

            float running_sum = 0.0f;
            int window_left = 0;
            int window_right = 0;       // one past the last point in the window

            for (int i = 0; i < number_of_errors; i++)
            {
                int want_left = i - NEIGHBORHOOD;
                int want_right = i + NEIGHBORHOOD + 1;

                if (want_left < 0)
                {
                    want_left = 0;
                }

                if (want_right > number_of_errors)
                {
                    want_right = number_of_errors;
                }

                while (window_right < want_right)
                {
                    running_sum = running_sum + prediction_errors[window_right];
                    window_right = window_right + 1;
                }

                while (window_left < want_left)
                {
                    running_sum = running_sum - prediction_errors[window_left];
                    window_left = window_left + 1;
                }

                float local_mean = running_sum / (window_right - window_left);
                local_deviations.push_back(prediction_errors[i] - local_mean);
            }

            // normal noise level of the deviations (skip the warm up)
            float squared_sum = 0.0f;
            int counted = 0;

            for (int i = WARM_UP; i < number_of_errors; i++)
            {
                squared_sum = squared_sum + local_deviations[i] * local_deviations[i];
                counted = counted + 1;
            }

            float deviation_std = sqrt(squared_sum / counted);

            // find the largest deviation and where it happened
            float worst_deviation = 0.0f;
            worst_point = 0;

            for (int i = WARM_UP; i < number_of_errors; i++)
            {
                float deviation = local_deviations[i];

                if (deviation < 0.0f)
                {
                    deviation = -deviation;
                }

                if (deviation > worst_deviation)
                {
                    worst_deviation = deviation;
                    worst_point = i + 1;    // error i belongs to data point i + 1
                }
            }

            return worst_deviation / deviation_std;
        }


//*****************   Main Program   *****************//

        int main(int argc, char *argv[])
        {
            const char *normal_directory = NORMAL_DIRECTORY;
            const char *anomaly_directory = ANOMALY_DIRECTORY;

            if (argc > 2)
            {
                normal_directory = argv[1];
                anomaly_directory = argv[2];
            }

            srand(42);      // same random starting weights every run


            // ─── Step 1: list the files ───
            std::vector<std::string> normal_files;
            std::vector<std::string> anomaly_files;

            if (list_csv_files(normal_directory, normal_files) != 0)
            {
                return 1;
            }

            if (list_csv_files(anomaly_directory, anomaly_files) != 0)
            {
                return 1;
            }

            printf("Found %d normal files and %d anomaly files.\n\n",
                    (int)normal_files.size(),
                    (int)anomaly_files.size());


            // ─── Step 2: cut training windows from the NORMAL files only ───
            printf("Building the training set from the normal files ...\n");

            std::vector< std::vector<float> > training_windows;

            for (int f = 0; f < (int)normal_files.size(); f++)
            {
                std::vector<float> power_values;
                std::vector<float> time_values;

                if (read_csv_file(normal_files[f].c_str(), power_values, time_values) != 0)
                {
                    return 1;
                }

                cut_training_windows(power_values, training_windows);
            }

            printf("Training set has %d windows.\n\n", (int)training_windows.size());


            // ─── Step 3: train ───
            printf("Training the LSTM ...\n");

            Lstm_Network network;

            network.train_network(  training_windows,
                                    WINDOW_SIZE,
                                    NUMBER_OF_EPOCHS,
                                    LEARNING_RATE);

            printf("Training done.\n\n");


            // ─── Steps 4 and 5: scan and score every file in both directories ───
            int normal_called_normal = 0;
            int anomaly_called_anomaly = 0;

            printf("─────────────────  NORMAL DIRECTORY  ─────────────────\n");

            for (int f = 0; f < (int)normal_files.size(); f++)
            {
                std::vector<float> power_values;
                std::vector<float> time_values;
                std::vector<float> prediction_errors;

                read_csv_file(normal_files[f].c_str(), power_values, time_values);
                network.scan_file_errors(power_values, prediction_errors);

                int worst_point = 0;
                float score = score_file(prediction_errors, worst_point);

                if (score > THRESHOLD_SIGMAS)
                {
                    printf("%-50s  score %6.1f   ANOMALY at point %d (time %.6f)\n",
                            normal_files[f].c_str(), score, worst_point, time_values[worst_point]);
                }
                else
                {
                    printf("%-50s  score %6.1f   normal\n", normal_files[f].c_str(), score);
                    normal_called_normal = normal_called_normal + 1;
                }
            }

            printf("\n─────────────────  ANOMALY DIRECTORY  ─────────────────\n");

            for (int f = 0; f < (int)anomaly_files.size(); f++)
            {
                std::vector<float> power_values;
                std::vector<float> time_values;
                std::vector<float> prediction_errors;

                read_csv_file(anomaly_files[f].c_str(), power_values, time_values);
                network.scan_file_errors(power_values, prediction_errors);

                int worst_point = 0;
                float score = score_file(prediction_errors, worst_point);

                if (score > THRESHOLD_SIGMAS)
                {
                    printf("%-50s  score %6.1f   ANOMALY at point %d (time %.6f)\n",
                            anomaly_files[f].c_str(), score, worst_point, time_values[worst_point]);
                    anomaly_called_anomaly = anomaly_called_anomaly + 1;
                }
                else
                {
                    printf("%-50s  score %6.1f   normal\n", anomaly_files[f].c_str(), score);
                }
            }


            // ─── Summary ───
            printf("\n─────────────────  SUMMARY  ─────────────────\n");
            printf("Normal files called normal:    %d out of %d\n",
                    normal_called_normal, (int)normal_files.size());
            printf("Anomaly files called anomaly:  %d out of %d\n",
                    anomaly_called_anomaly, (int)anomaly_files.size());

            return 0;
        }
