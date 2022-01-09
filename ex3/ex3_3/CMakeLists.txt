cmake_minimum_required(VERSION 3.18)
project(uebung3_3 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(uebung3_3 trimp.cu)

set_target_properties(uebung3_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
