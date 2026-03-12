/// @file csvTools.h
/// @brief Header file for CSV parsing and printing utilities in the SimpleNN project.
///
/// Declares functions for:
/// - Parsing individual CSV files into 2D nested lists of floats.
/// - Printing single or multiple parsed CSV tables.
/// - Loading all CSVs from a directory into a 3D structure.
///
/// These utilities form the data ingestion layer for regression datasets,
/// supporting header skipping, basic numeric parsing, and structured output.
///
/// @note Designed for simple numeric CSVs (no quoted fields, no escaped commas).
///       Assumes consistent row lengths and float-convertible values.
///       For advanced CSV handling (quotes, strings, large files), consider third-party libraries.
///
/// @author Gary
/// @version 0.1.0
/// @date February 2026
///
/// @see csvTools.cpp for implementations and detailed parsing logic.

#ifndef __CSVTOOLS_H__
#define __CSVTOOLS_H__

#include <iostream>      ///< For console output (debugging/printing)
#include <fstream>       ///< For file input streams
#include <list>          ///< For dynamic nested storage (rows/columns/files)
#include <string>        ///< For file paths and extensions
#include <cstdlib>       ///< For strtof (string to float conversion)
#include <cstring>       ///< For memset (buffer clearing)
#include <filesystem>    ///< For directory scanning and existence checks

/// @brief Namespace alias for std::filesystem to improve readability.
namespace fs = std::filesystem;

/// @brief Parses a single CSV file into a 2D list of floats (rows → columns).
///
/// Opens the file, skips the header line, and reads data char-by-char.
/// Handles comma/newline separators and basic float conversion.
///
/// @param filename Path to the CSV file.
/// @return 2D list representing parsed rows and columns.
///         Empty on file error or no data.
///
/// @note See csvTools.cpp for full implementation details, including:
///       - Buffer management and overflow handling.
///       - Invalid value skipping.
///       - Final field processing.
///
/// @warning Limited to numeric data; non-float values are skipped with diagnostics.
std::list<std::list<float>> csv2list(std::string filename);

/// @brief Prints a single parsed CSV table to stdout.
///
/// @param csv Reference to 2D list (rows → columns) to print.
///
/// @note Const-correct iteration; formats with row numbers and space separation.
///       Suitable for verification/debugging of loaded data.
void printCSV(std::list<std::list<float>> &csv);

/// @brief Loads all CSV files from a directory into a 3D list structure.
///
/// Scans the directory recursively (via std::filesystem), filters by .csv/.CSV extension,
/// and parses each file using csv2list.
///
/// @param dir_path Path to directory containing CSV files.
/// @return 3D list: files → rows → columns (floats).
///         Empty if directory invalid or no matching files.
///
/// @note Adds tables even if parsing yields empty rows (configurable in implementation).
///       Provides console feedback during loading.
///
/// @see csv2list() for per-file parsing details.
std::list<std::list<std::list<float>>> load_all_csv_from_directory(const std::string& dir_path);

/// @brief Prints all loaded CSV tables with clear separation.
///
/// @param all_csv_data Const reference to 3D list (files → rows → columns).
///
/// @note Includes file headers, row numbering, and blank lines between tables.
///       Ideal for comprehensive data verification.
void printCSV_all(const std::list<std::list<std::list<float>>>& all_csv_data);




std::list<std::list<float>> single_list(std::list<std::list<std::list<float>>> data, int file_idx);

#endif // __CSVTOOLS_H__