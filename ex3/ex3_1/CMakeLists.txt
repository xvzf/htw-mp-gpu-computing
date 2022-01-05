cmake_minimum_required(VERSION 3.18)
project(uebung3_1 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(uebung3_1 main.cu)

set_target_properties(uebung3_1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
