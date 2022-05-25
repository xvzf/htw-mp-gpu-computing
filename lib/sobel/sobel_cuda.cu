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


__global__ void sobelKernel(uint8_t *in_first_row, uint8_t *in_second_row, uint8_t *in_third_row, uint8_t *out, uintmax_t out_size_x)
{
    uintmax_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < out_size_x)
    {
        uintmax_t y = idx / out_size_x;
        uintmax_t x = idx % out_size_x;
        out[y * out_size_x + x] = sobel_combined_op_data_three_rows(in_first_row, in_second_row, in_third_row, x);
    }
}

void sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset, int last_call)
{
    static uint8_t *device_in_first_row;
    static uint8_t *device_in_second_row;
    static uint8_t *device_in_third_row;
    if(offset == 1){
        gpuErrchk(cudaMalloc((void **)&device_in_first_row, in_img->size_x * sizeof(uint8_t)));
        gpuErrchk(cudaMalloc((void **)&device_in_second_row, in_img->size_x * sizeof(uint8_t)));
        gpuErrchk(cudaMalloc((void **)&device_in_third_row, in_img->size_x * sizeof(uint8_t)));

        gpuErrchk(cudaMemcpy(device_in_first_row, in_img->data, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(device_in_second_row, in_img->data + in_img->size_x, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(device_in_third_row, in_img->data + in_img->size_x * 2, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
    }
    else{
        gpuErrchk(cudaMemcpy(device_in_first_row, in_img->data + in_img->size_x * (2+offset), sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
        uint8_t *tmp = device_in_first_row;
        device_in_first_row = device_in_second_row;
        device_in_second_row = device_in_third_row;
        device_in_third_row = tmp;
    }
    static bool device_out_initialized = false;
    if(!device_out_initialized) {
        gpuErrchk(cudaMalloc((void **) &device_out, out_img->size_x * sizeof(uint8_t)));
        device_out_initialized = true;
    }

    static bool maps_transferred = false;
    if(!maps_transferred){
        gpuErrchk(cudaMemcpyToSymbol(constant_maps, &constant_maps_host, MAP_SIZE * sizeof(uint8_t)));
        maps_transferred = true;
    }
    static uintmax_t N = out_img->size_x;
    static uintmax_t block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sobelKernel<<<block_count, THREADS_PER_BLOCK>>>(device_in_first_row, device_in_second_row, device_in_third_row, device_out, out_img->size_x);
    static uint8_t *pixel_line;
    if(offset != in_img->size_y - 3){

        pixel_line = read_pixel_line(in_img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
            return ;
        }
        memcpy(in_img->data + (offset + 4) * in_img->size_x, pixel_line, in_img->size_x);
    }
     gpuErrchk(cudaPeekAtLastError());
     gpuErrchk(cudaDeviceSynchronize());
     gpuErrchk(cudaMemcpy(out_img->data + offset * out_img->size_x, device_out, sizeof(uint8_t) * out_img->size_x, cudaMemcpyDeviceToHost));
    if(last_call){
        printf("Last call\n");
        cudaFree(device_in_first_row);
        cudaFree(device_in_second_row);
        cudaFree(device_in_third_row);
        cudaFree(device_out);
    }
}