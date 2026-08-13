# ──────────────────────────────────────────────────────────────────────────────
# CMakeLists.txt — same structure as the LSTM_unit project
#
# Sources are globbed recursively out of src/, and every directory under
# includes/ is added to the include path. That means you can add a new layer by
# dropping two files into the tree and rebuilding, without editing this file.
# ──────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(TransformerLLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation matters here in a way it did not for a single LSTM unit — the
# naive matrix multiply is much faster with -O2. Release is the default so a
# plain "cmake .." does the right thing.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Recursively find all source files
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cc
)

add_executable(main
    main.cpp
    ${SRC_FILES}
)

# === Properly add all include directories (including subfolders) ===
target_include_directories(main PRIVATE includes)

# Get all items under includes/ recursively
file(GLOB_RECURSE ALL_ITEMS LIST_DIRECTORIES true includes/*)

# Filter to keep ONLY directories
set(DIR_LIST "")
foreach(item IN LISTS ALL_ITEMS)
    if(IS_DIRECTORY "${item}")
        list(APPEND DIR_LIST "${item}")
    endif()
endforeach()

target_include_directories(main PRIVATE ${DIR_LIST})
