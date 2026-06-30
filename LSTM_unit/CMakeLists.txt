cmake_minimum_required(VERSION 3.16)
project(LinearRegression LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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