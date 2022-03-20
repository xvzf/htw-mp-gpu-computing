#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include "sobel.h"
#include "sobel_cuda.cuh"
#include "lib/img/types.h"
#include "lib/img/operations.h"

#define THREADS_PER_BLOCK (256)

__global__ void sobelKernel(uint8_t *in, uint8_t *out, uintmax_t in_size_x, uintmax_t in_size_y, uintmax_t out_size_x, uintmax_t out_size_y)
{
    uintmax_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < out_size_x * out_size_y)
    {
        uintmax_t y = idx / out_size_x;
        uintmax_t x = idx % out_size_x;
        out[y * out_size_x + x] = sobel_combined_op_data2(in, x, y, in_size_x, in_size_y);
    }
}

int sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset)
{
    int err = 0;

    uint8_t *device_in, *device_out, *device_maps;
    err += gpuErrchk(cudaMalloc((void **)&device_in, in_img->size_x * in_img->size_y * sizeof(uint8_t)));
    err += gpuErrchk(cudaMalloc((void **)&device_out, out_img->size_x * out_img->size_y * sizeof(uint8_t)));
    err += gpuErrchk(cudaMalloc((void **)&device_maps, MAP_SIZE * sizeof(uint8_t)));
    err += gpuErrchk(cudaMemcpy(device_in, in_img->data, sizeof(uint8_t) * in_img->size_x * in_img->size_y, cudaMemcpyHostToDevice));
    //uintmax_t N = out_img->size_x * out_img->size_y;
    uint8_t host_maps [MAP_SIZE];
    for(int i = 0; i < MAP_SIZE;++i){
        host_maps[i] = sqrt(i);
    }
    err+= gpuErrchk(cudaMemcpyToSymbol(constant_maps, &host_maps, MAP_SIZE*sizeof(uint8_t)));
    err += gpuErrchk(cudaMemcpy(device_in, in_img->data + offset * in_img->size_x, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
    uintmax_t N = out_img->size_x;

    // Run kernel
    sobelKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_in, device_out, in_img->size_x, in_img->size_y, out_img->size_x, out_img->size_y);

    err += gpuErrchk(cudaPeekAtLastError());
    err += gpuErrchk(cudaDeviceSynchronize());
    err += gpuErrchk(cudaMemcpy(out_img->data + offset * in_img->size_x, device_out, sizeof(uint8_t) * out_img->size_x, cudaMemcpyDeviceToHost));

    // Free up internal resources
    cudaFree(device_in);
    cudaFree(device_out);

    return err;
}
