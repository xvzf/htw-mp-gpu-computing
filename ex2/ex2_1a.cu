#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (2048 * 2048)       // N
#define THREADS_PER_BLOCK 128 // B

// random_floats fills an array of floats; not that random :-)
void random_floats(float *a, int array_size)
{
    for (int i = 0; i < array_size; i++)
    {
        a[i] = (float)i;
    }
}

// add is the default vector add
__global__ void add(float *a, float *b, float *c, int n)
{

    int idx = blockIdx.x + threadIdx.x * gridDim.x;

    // Not needed, just for safety!
    c[idx] = a[idx] + b[idx];
}

// main is the main entrypoint.
int main(void)
{
    float *a, *b, *c;       // host copies of a, b, c
    float *d_a, *d_b, *d_c; // device copies of a, b, c
    float size = N * sizeof(float);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for cpu
    a = (float *)malloc(size);
    random_floats(a, N);
    b = (float *)malloc(size);
    random_floats(b, N);
    c = (float *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    add<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    printf("%f\n", c[N - 1]);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
