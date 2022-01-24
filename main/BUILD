cc_binary(
    name = "sobel_seq",
    srcs = ["main.c"],
    deps = [
        "//lib/img",
        "//lib/sobel:seq",
    ],
)

cc_binary(
    name = "sobel_omp",
    srcs = ["main.c"],
    deps = [
        "//lib/img",
        "//lib/sobel:omp",
    ],
)

cc_binary(
    name = "sobel_cuda",
    srcs = ["main.c"],
    deps = [
        "//lib/img",
        "//lib/sobel:cuda",
    ],
)
