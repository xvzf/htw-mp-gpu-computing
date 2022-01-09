cmake_minimum_required(VERSION 3.18)
project(uebung3_2 CXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(uebung3_2 trimp.cpp)

