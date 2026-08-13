# ──────────────────────────────────────────────────────────────────────────────
# CMakeLists.txt — recurses through every subdirectory of src/ and includes/
#
# Same structure as LSTM_unit, tightened up:
#   - source files:  every .cpp / .cc under src/, at any depth
#   - header search: every directory under includes/, at any depth, PLUS
#                     includes/ itself
#
# That means the project layout is free to grow — a new folder like
# includes/Layers/Attention/ with a matching src/Layers/Attention/ is picked up
# automatically the next time you run `cmake ..`. You never edit this file to
# add a new class, only to change a build setting.
# ──────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(TransformerLLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default — the naive matrix multiply needs -O2 to run at a
# reasonable speed, and a plain "cmake .." should just do the right thing.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# ────────────────────────────────────────────────────────────────────────────
# SOURCE FILES — every .cpp / .cc anywhere under src/
#
# CONFIGURE_DEPENDS makes CMake re-scan the src/ tree on every build, not just
# when you re-run cmake by hand. That is what lets you drop a new .cpp into a
# new subfolder and have `cmake --build build` pick it up on its own.
# (A few CMake/generator combinations only re-check this at configure time —
# if a brand-new file is ever silently skipped, deleting build/ and
# reconfiguring from scratch always picks it up.)
# ────────────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cc
)

add_executable(main
    main.cpp
    ${SRC_FILES}
)


# ────────────────────────────────────────────────────────────────────────────
# HEADER SEARCH PATHS — includes/ itself, plus every subdirectory under it
#
# We add includes/ directly (for the common case of a header sitting right at
# the top level) and then walk the tree collecting every directory we find, so
# "#include "some_header.h"" resolves no matter how deep some_header.h lives.
# ────────────────────────────────────────────────────────────────────────────
target_include_directories(main PRIVATE includes)

file(GLOB_RECURSE ALL_INCLUDE_ITEMS CONFIGURE_DEPENDS LIST_DIRECTORIES true includes/*)

set(INCLUDE_DIR_LIST "")
foreach(item IN LISTS ALL_INCLUDE_ITEMS)
    if(IS_DIRECTORY "${item}")
        list(APPEND INCLUDE_DIR_LIST "${item}")
    endif()
endforeach()

target_include_directories(main PRIVATE ${INCLUDE_DIR_LIST})


# ────────────────────────────────────────────────────────────────────────────
# REPORT WHAT WAS FOUND
#
# Printed every time you configure (cmake ..), so a file that silently failed
# to get picked up — like the stale activation_functions.cpp that started this
# whole investigation — is visible immediately instead of showing up later as
# a mysterious linker error.
# ────────────────────────────────────────────────────────────────────────────
message(STATUS "── Transformer LLM build ──")
message(STATUS "Source files found (${CMAKE_CURRENT_SOURCE_DIR}/src/**):")
foreach(src_file IN LISTS SRC_FILES)
    message(STATUS "    ${src_file}")
endforeach()

message(STATUS "Header search directories:")
message(STATUS "    ${CMAKE_CURRENT_SOURCE_DIR}/includes")
foreach(inc_dir IN LISTS INCLUDE_DIR_LIST)
    message(STATUS "    ${inc_dir}")
endforeach()
