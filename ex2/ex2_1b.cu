#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define ROWS 3 // M
#define COLS 4 // N

void print_matrix(float *a, int m, int n)
{

    printf("(\n");
    for (int i = 0; i < m; i++)
    {
        printf("\t");
        for (int j = 0; j < n; j++)
        {
            int idx = i * m + j;
            printf("[idx=%d] %0.0f\t", idx, a[idx]);
        }
        printf("\n");
    }
    printf(")\n");
}

// add is the default vector add
__global__ void idx_map_odd_even(float *c)
{
    int idx;
    if (threadIdx.x < (gridDim.x / 2))
    {
        // blockDim.x happens to be the matrix row count
        idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    }
    else
    {
        idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    }

    c[idx] = (float)idx;
}

// main is the main entrypoint.
int main(void)
{
    float *c;
    float *d_c;
    float size = ROWS * COLS * sizeof(float);

    cudaMalloc((void **)&d_c, size);

    c = (float *)malloc(size);

    // Launch add() kernel on GPU
    idx_map_odd_even<<<ROWS, COLS>>>(d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Show matrix with indices
    print_matrix(c, ROWS, COLS);

    // Cleanup
    cudaFree(d_c);
    free(c);
    return 0;
}
