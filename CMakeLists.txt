cmake_minimum_required(VERSION 3.16)
project(LinearRegression LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically adds -Iincludes
include_directories(includes)

add_executable(main
    main.cpp
    src/csvTools.cpp
    # Add future .cpp files here
)