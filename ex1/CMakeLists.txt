cmake_minimum_required(VERSION 3.12)
project(openmptest C)

set(CMAKE_XCODE_GENERATE_SCHEME ON)
set(CMAKE_XCODE_SCHEME_ADDRESS_SANITIZER ON)
set(CMAKE_XCODE_SCHEME_ADDRESS_SANITIZER_USE_AFTER_RETURN ON)


add_executable(ex1 ex1.c)

find_package(OpenMP REQUIRED)
target_link_libraries(ex1 PRIVATE OpenMP::OpenMP_C)
