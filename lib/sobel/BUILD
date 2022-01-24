load("@rules_cuda//cuda:defs.bzl", "cuda_library", "requires_cuda_enabled")

cc_library(
    name = "seq",
    srcs = [ "sobel_seq.c" ],
    hdrs = ["sobel.h"],
    visibility = ["//main:__pkg__"],
    deps = [ "//lib/img" ],
)

cc_library(
    name = "omp",
    srcs = [ "sobel_omp.c" ],
    hdrs = ["sobel.h"],
    visibility = ["//main:__pkg__"],
    deps = [ "//lib/img" ],
)

cuda_library(
    name = "cuda",
    srcs = ["sobel_cuda.cu"],
    hdrs = [
        "sobel.h",
        "sobel_cuda.cuh",
    ],
    visibility = ["//main:__pkg__"],
    deps = [ "//lib/img" ],
    target_compatible_with = requires_cuda_enabled(),
)