#ifndef __SOBEL_CUDA_H__
#define __SOBEL_CUDA_H__

// error check with nice line numbers and file name shamelessly stolen from https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) \
    gpuAssert((ans), __FILE__, __LINE__);

static inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#endif