/// @file csvTools.cpp
/// @brief Implementation of CSV parsing and printing utilities for the SimpleNN project.
///
/// This file contains the definitions for functions declared in csvTools.h.
/// It provides:
/// - Low-level char-by-char CSV parsing into nested std::list structures.
/// - Header skipping and basic error handling.
/// - Directory scanning for multiple CSV files.
/// - Structured printing for verification and debugging.
///
/// @note The parser is designed for simple numeric CSVs (no quoted fields, no escaped commas).
/// It uses fixed-size buffering (100 chars per field) for simplicity—suitable for small datasets.
/// For production use, consider libraries like rapidcsv for robustness.
///
/// @see csvTools.h for function declarations and high-level API documentation.

#include "csvTools.h"

/// @brief Parses a single CSV file into a 2D list of floats (rows → columns).
///
/// Opens the file, skips the first line (assumed header), then reads char-by-char.
/// Fields are separated by commas, rows by newlines.
/// Uses strtof for conversion with validation.
///
/// @param filename Path to the CSV file (std::string).
/// @return std::list<std::list<float>> representing rows of parsed float values.
///         Empty list on file error or no data rows.
///
/// @note
/// - Empty fields (e.g., consecutive commas) are skipped (treated as no value).
/// - Invalid numbers (non-numeric characters) are skipped with a diagnostic message.
/// - Fixed buffer size (100 chars) — fields longer than 99 chars are truncated.
/// - Last field on final line handled even without trailing newline.
///
/// @warning Does not support quoted fields, escaped commas, or string data.
///          Assumes all values are numeric floats.
///
/// @details Parsing logic:
/// - Accumulates characters into a buffer until ',' or '\n'.
/// - Null-terminates and parses with strtof.
/// - Validates full consumption (endptr check).
/// - Uses emplace_back/push_back for efficient list construction.
std::list<std::list<float>> csv2list(std::string filename)
{
    std::list<std::list<float>> csv;               ///< Outer list (rows), inner lists (columns)
    char buffer[100];                              ///< Fixed buffer for field accumulation
    int string_index = 0;                          ///< Index in buffer for current field

    std::ifstream file(filename);                  ///< Input file stream

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return csv;  // Return empty on failure
    }

    char letter;  ///< Current character read from stream

    // Skip header line (read until '\n' inclusive)
    while (file.get(letter))
    {
        if (letter == '\n')
        {
            break;
        }
    }

    std::cout << "Header skipped. Starting data..." << std::endl;

    // Main parsing loop: Read char-by-char until EOF
    while (file.get(letter))
    {
        // Field or row separator?
        if (letter == ',' || letter == '\n')
        {
            // Process accumulated field if non-empty
            if (string_index > 0)
            {
                buffer[string_index] = '\0';           // Null-terminate for strtof
                char *endptr;
                float value = strtof(buffer, &endptr);

                // Full parse validation
                if (endptr == buffer + string_index)
                {
                    // Ensure current row exists
                    if (csv.empty())
                    {
                        csv.emplace_back();            // New empty row
                    }
                    csv.back().push_back(value);       // Add value to current row
                }
                else
                {
                    std::cout << "Invalid number skipped: '" << buffer << "'" << std::endl;
                }
            }

            // Reset buffer for next field
            std::memset(buffer, 0, sizeof(buffer));
            string_index = 0;

            // Newline → start new row
            if (letter == '\n')
            {
                csv.emplace_back();
            }
        }
        else
        {
            // Regular character — accumulate if buffer not full
            if (string_index < sizeof(buffer) - 1)
            {
                buffer[string_index++] = letter;
            }
            // Silent truncation for very long fields
        }
    }

    // Handle final field without trailing separator
    if (string_index > 0)
    {
        buffer[string_index] = '\0';
        char* endptr;
        float value = strtof(buffer, &endptr);
        if (endptr == buffer + string_index)
        {
            if (csv.empty())
            {
                csv.emplace_back();
            }
            csv.back().push_back(value);
        }
    }

    file.close();
    return csv;
}

/// @brief Prints a single parsed CSV table to stdout.
///
/// @param csv Reference to 2D list (rows → columns) to print.
///
/// @note Const-correct iteration for safety.
///       Formats as space-separated values with row numbers.
void printCSV(std::list<std::list<float>> &csv)
{
    std::cout << "\nLoaded " << csv.size() << " rows:" << std::endl;

    int row_num = 0;
    for (auto row_it = csv.begin(); row_it != csv.end(); ++row_it)
    {
        const auto& current_row = *row_it;
        std::cout << "Row " << row_num++ << ": ";

        for (auto val_it = current_row.begin(); val_it != current_row.end(); ++val_it)
        {
            std::cout << *val_it << " ";
        }
        std::cout << std::endl;
    }
}

/// @brief Loads all CSV files from a directory into a 3D list structure.
///
/// Scans directory, filters .csv/.CSV files, parses each with csv2list.
///
/// @param dir_path Path to directory containing CSVs.
/// @return 3D list: files → rows → columns (floats).
///         Empty if directory invalid or no CSVs.
///
/// @note Uses std::filesystem for portable directory iteration.
///       Adds tables even if empty (configurable via condition).
std::list<std::list<std::list<float>>> load_all_csv_from_directory(const std::string& dir_path)
{
    std::list<std::list<std::list<float>>> all_csv_data;

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path))
    {
        std::cerr << "Error: Directory does not exist or is not a directory: " << dir_path << std::endl;
        return all_csv_data;
    }

    std::cout << "Scanning directory: " << dir_path << std::endl;

    for (const auto& entry : fs::directory_iterator(dir_path))
    {
        if (entry.is_regular_file())
        {
            std::string file_path = entry.path().string();
            std::string extension = entry.path().extension().string();

            if (extension == ".csv" || extension == ".CSV")
            {
                std::cout << "Found CSV file: " << file_path << std::endl;

                std::list<std::list<float>> table = csv2list(file_path);

                // Always add (even empty tables) — change to !table.empty() to skip
                all_csv_data.push_back(table);
                std::cout << "Loaded " << table.size() << " rows from " << file_path << std::endl;
            }
        }
    }

    std::cout << "Finished loading. Total CSV files processed: " << all_csv_data.size() << std::endl;
    return all_csv_data;
}

/// @brief Prints all loaded CSV tables with file separators and row numbering.
///
/// @param all_csv_data Const reference to 3D list (files → rows → columns).
///
/// @note Provides clear visual separation between files for debugging/verification.
void printCSV_all(const std::list<std::list<std::list<float>>>& all_csv_data)
{
    std::cout << "\nTotal number of CSV files loaded: " << all_csv_data.size() << "\n\n";

    int file_num = 0;
    for (auto file_it = all_csv_data.begin(); file_it != all_csv_data.end(); ++file_it)
    {
        const auto& current_table = *file_it;

        std::cout << "=== CSV File " << file_num++ << " (" << current_table.size() << " rows) ===\n";

        int row_num = 0;
        for (auto row_it = current_table.begin(); row_it != current_table.end(); ++row_it)
        {
            const auto& current_row = *row_it;

            std::cout << "Row " << row_num++ << ": ";

            for (auto val_it = current_row.begin(); val_it != current_row.end(); ++val_it)
            {
                std::cout << *val_it << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;  // Blank line between files
    }
}


std::list<std::list<float>> single_list(std::list<std::list<std::list<float>>> data, int file_idx)
{
    std::list<std::list<float>> single_list;
    std::list<std::list<std::list<float>>>::const_iterator file_it = data.begin();

    std::advance(file_it, file_idx);  // Move iterator to the desired file index
    
    if (file_it != data.end()) {
        single_list = *file_it;
    } else {
        std::cerr << "Error: File index out of range." << std::endl;
    }
    return single_list;
}







