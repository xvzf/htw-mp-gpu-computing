# htw-mp-gpu-computing project

![example workflow](https://github.com/xvzf/htw-mp-gpu-computing/actions/workflows/workflow.yml/badge.svg)


## Tooling
[Bazel](https://bazel.build) is required, checkout `.github/workflows/workflow.yml` as reference for installing.
In addition, we're prefering the [LLVM clang compiler](https://clang.llvm.org) (version 13).

We're using Tensorflows cuda library for bazel ([ref](https://github.com/tensorflow/runtime/tree/master/third_party/rules_cuda))
