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


__global__ void sobelKernel(uint8_t *in_first_row, uint8_t *in_second_row, uint8_t *in_third_row, uint8_t *out, uintmax_t in_size_x, uintmax_t in_size_y, uintmax_t out_size_x, uintmax_t out_size_y)
{
    uintmax_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < out_size_x)
    {
        uintmax_t y = idx / out_size_x;
        uintmax_t x = idx % out_size_x;
       // printf("%u ",y * out_size_x + x);
        out[y * out_size_x + x] = sobel_combined_op_data_three_rows(in_first_row, in_second_row, in_third_row, x, y, in_size_x, in_size_y);
    }
}

int write_to_out_img(ppm_image *out_img, intmax_t offset, uint8_t *_device_out){
    printf("write_out\n");
    printf("\n%p %p\n", device_out, _device_out);
    printf("%p %p\n", out_img, out_img->data);
   // gpuErrchk(cudaPeekAtLastError());
   // gpuErrchk(cudaDeviceSynchronize());
    printf("offset: %u \n", offset);
   // gpuErrchk(cudaMemcpy(out_img->data + offset * out_img->size_x, _device_out, sizeof(uint8_t) * out_img->size_x, cudaMemcpyDeviceToHost));
    return 0;
}

uint8_t *sobel(ppm_image *in_img, ppm_image *out_img, intmax_t offset, uint8_t **return_device_out)
{
    int err = 0;

    static uint8_t *device_in_first_row;
    static uint8_t *device_in_second_row;
    static uint8_t *device_in_third_row;
    if(offset == 0){
        err += gpuErrchk(cudaMalloc((void **)&device_in_first_row, in_img->size_x * sizeof(uint8_t)));
        err += gpuErrchk(cudaMalloc((void **)&device_in_second_row, in_img->size_x * sizeof(uint8_t)));
        err += gpuErrchk(cudaMalloc((void **)&device_in_third_row, in_img->size_x * sizeof(uint8_t)));

        err += gpuErrchk(cudaMemcpy(device_in_first_row, in_img->data, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
        err += gpuErrchk(cudaMemcpy(device_in_second_row, in_img->data + in_img->size_x, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
        err += gpuErrchk(cudaMemcpy(device_in_second_row, in_img->data + in_img->size_x * 2, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
    }
    else{
        uint8_t *tmp = device_in_first_row;
        device_in_first_row = device_in_second_row;
        device_in_second_row = device_in_third_row;
        device_in_third_row = tmp;
        err += gpuErrchk(cudaMemcpy(device_in_second_row, in_img->data + in_img->size_x * (2+offset), sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
    }
    static bool device_out_initialized = false;
    if(!device_out_initialized) {
        err += gpuErrchk(cudaMalloc((void **) &device_out, out_img->size_x * sizeof(uint8_t)));
        device_out_initialized = true;
    }

    uint8_t *device_maps;
    static bool maps_transferred = false;
    if(!maps_transferred){
        err += gpuErrchk(cudaMalloc((void **)&device_maps, MAP_SIZE * sizeof(uint8_t)));
        err+= gpuErrchk(cudaMemcpyToSymbol(constant_maps, &constant_maps_host, MAP_SIZE*sizeof(uint8_t)));
        maps_transferred = true;
    }
    uintmax_t N = out_img->size_x;

    sobelKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_in_first_row, device_in_second_row, device_in_third_row, device_out, in_img->size_x, in_img->size_y, out_img->size_x, out_img->size_y);
   // printf("\nkernel: %u done, device_out: %p\n", offset, device_out);

    //printf("s %p %p\n", return_device_out, *return_device_out);
    *return_device_out = device_out;
    if(offset != in_img->size_y - 3){

        uint8_t *pixel_line = read_pixel_line(in_img);
        if(pixel_line == NULL)
        {
            fprintf(stderr, "'%s' couldn't be loaded", in_img->filename);
            return NULL;
        }
        memcpy(in_img->data + (offset + 4) * in_img->size_x, pixel_line, in_img->size_x);
        free(pixel_line);
    }
    //printf("s %p %p\n", return_device_out, *return_device_out);

    //write_to_out_img(out_img, offset, device_out);
     err += gpuErrchk(cudaPeekAtLastError());
     err += gpuErrchk(cudaDeviceSynchronize());
     err += gpuErrchk(cudaMemcpy(out_img->data + offset * out_img->size_x, device_out, sizeof(uint8_t) * out_img->size_x, cudaMemcpyDeviceToHost));
////
    // Free up internal resources
    //cudaFree(device_in);
    //cudaFree(device_out);

    return device_out;
}


int sobel2(ppm_image *in_img, ppm_image *out_img, intmax_t offset)
{
    int err = 0;
/*
    uint8_t *device_in, *device_out, *device_maps;
    err += gpuErrchk(cudaMalloc((void **)&device_in, in_img->size_x * sizeof(uint8_t)));
    err += gpuErrchk(cudaMalloc((void **)&device_out, out_img->size_x * sizeof(uint8_t)));
    err += gpuErrchk(cudaMalloc((void **)&device_maps, MAP_SIZE * sizeof(uint8_t)));
    err += gpuErrchk(cudaMemcpy(device_in, in_img->data, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
    //uintmax_t N = out_img->size_x * out_img->size_y;
    if(!maps_transferred){
        err+= gpuErrchk(cudaMemcpyToSymbol(constant_maps, &constant_maps_host, MAP_SIZE*sizeof(uint8_t)));
        maps_transferred = true;
    }
    err += gpuErrchk(cudaMemcpy(device_in, in_img->data + offset * in_img->size_x, sizeof(uint8_t) * in_img->size_x, cudaMemcpyHostToDevice));
    uintmax_t N = out_img->size_x;

    // Run kernel
    printf("offset: %u size_x: %u size_y: %u zeilenanfang: %u\n", offset, in_img->size_x, in_img->size_y, offset * in_img->size_x);
    for(int i = 0; i < out_img->size_x; ++i){
        printf("%3u %u  ", in_img->data[i+offset*in_img->size_x], i+offset*in_img->size_x);
    }
    printf("\n");
    sobelKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_in, device_out, in_img->size_x, in_img->size_y, out_img->size_x, out_img->size_y);
    printf("\nkernel: %u done\n", offset);
    err += gpuErrchk(cudaPeekAtLastError());
    err += gpuErrchk(cudaDeviceSynchronize());
    err += gpuErrchk(cudaMemcpy(out_img->data + offset * in_img->size_x, device_out, sizeof(uint8_t) * out_img->size_x, cudaMemcpyDeviceToHost));

    // Free up internal resources
    cudaFree(device_in);
    cudaFree(device_out);*/

    return err;
}
