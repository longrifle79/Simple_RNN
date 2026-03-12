/// @file main.cpp
/// @brief Program entry point for the SimpleNN linear regression CSV data loader.
///
/// This file contains the `main()` function, which serves as the executable's starting point.
/// It demonstrates the usage of the CSV utilities from csvTools.h/cpp:
/// - Specifies a hard-coded directory containing CSV datasets.
/// - Loads all CSV files into a 3D nested list structure (files → rows → float values).
/// - Prints the total number of files loaded.
/// - Outputs the full contents of all tables for verification.
///
/// The program is designed as a simple command-line tool for quick data inspection
/// and validation during development of regression models in the SimpleNN project.
///
/// @note
/// - The CSV directory path is currently hard-coded for convenience.
///   In a production or reusable tool, this should be passed as a command-line argument
///   (e.g., via argc/argv) or configured via a file/environment variable.
/// - No error handling beyond what csvTools provides—assumes valid numeric CSVs.
/// - Output is to stdout; suitable for piping or redirection (e.g., `./main > output.txt`).
///
/// @warning
/// - Relies on csvTools for parsing—see its documentation for limitations
///   (e.g., no support for quoted fields, fixed buffer size).
/// - Hard-coded path ties the executable to a specific filesystem location.
///
/// @author Gary
/// @version 0.1.0
/// @date February 2026
///
/// @see csvTools.h for the core CSV loading and printing API.
/// @see load_all_csv_from_directory() for directory scanning details.
/// @see printCSV_all() for structured output formatting.

#include <iostream>      ///< For std::cout (console output)
#include <fstream>       ///< For file streams (used indirectly via csvTools)
#include <list>          ///< For nested dynamic storage of CSV data
#include <string>        ///< For directory path string
#include <cstdlib>       ///< For general utilities (inherited from csvTools)
#include <cstring>       ///< For buffer operations (inherited from csvTools)
#include <filesystem>    ///< For directory existence checks (inherited from csvTools)
#include "csvTools.h"    ///< CSV parsing and printing utilities
#include "lsr.h"        ///< Linear regression utilities (potentially used later)

/// @brief Main program entry point.
///
/// Initializes the CSV loading process:
/// 1. Defines the target directory containing regression datasets.
/// 2. Calls load_all_csv_from_directory() to parse all .csv files.
/// 3. Reports the number of files successfully loaded.
/// 4. Prints the full contents of all parsed tables.
///
/// @return int Exit status: 0 on success (standard convention).
///
/// @note
/// - The 3D list structure represents: files → rows → columns (as floats).
/// - This simple driver is ideal for quick testing and debugging of the csvTools module.
/// - No command-line arguments are processed—extend with argc/argv for flexibility.
///
/// @details Execution flow:
/// - Hard-coded directory path for reproducibility during development.
/// - Relies on csvTools for all heavy lifting (parsing, validation, printing).
/// - Minimal error checking here—csvTools handles file/directory issues with diagnostics.
///
/// @warning Hard-coded absolute path—change to match your environment or make configurable.
int main()
{
    /// @brief Target directory containing CSV regression datasets.
    ///
    /// @note Update this path if your csv_data folder moves.
    ///       Consider making it a command-line parameter for portability.
    const std::string directory = "/home/gary/Desktop/Neural_Network_Study/Data/Normal";

    /// @brief Loaded 3D structure containing all parsed CSV data.
    ///
    /// Structure: files → rows → columns (floats).
    /// Empty if directory invalid or no CSVs found.
    std::list<std::list<std::list<float>>> all_data = load_all_csv_from_directory(directory);

    /// Report total number of CSV files processed.
    std::cout << "Total CSV files loaded: " << all_data.size() << std::endl;
      printCSV_all(all_data);

    std::cout << "Printing contents of a single file..." << std::endl;
    std::list<std::list<float>> single_file_data = single_list(all_data, 0);  // Get first file's data
    printCSV(single_file_data);

  
  float ls = least_squares(single_file_data, LEARNING_RATE);

  std::cout << "LEAST SQUARES IS: " << ls <<  std::endl;

    /// @return Program exit status (0 = success).
    return 0;
}