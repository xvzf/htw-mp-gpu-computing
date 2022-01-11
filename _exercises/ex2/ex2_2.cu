#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (16 * 16)         // N
#define THREADS_PER_BLOCK 4 // B

// random_floats fills an array of floats; not that random :-)
void random_floats(float *a, int array_size)
{
    for (int i = 0; i < array_size; i++)
    {
        a[i] = (float)i;
    }
}

__global__ void vecads_orig(float *a, float *b, float *c, int n)
{
    int idx = blockIdx.x + threadIdx.x * gridDim.x;
    if (idx < n)
        if (idx & 1)
            c[idx] = a[idx] + b[idx];
        else
            c[idx] = a[idx] - b[idx];
}

// Regularized version of vecads
__global__ void vecads_reg(float *a, float *b, float *c, int n)
{
    int idx = blockIdx.x + threadIdx.x * gridDim.x;
    if (idx < n)
    {
        // The if else condition can be simplified with no requirements to the value range.
        // (idx & 1) is 1 when the LSB is 1 and vice versa thus checking if the value is even or odd.
        // -> the injected therm `(2*(idx&1)-1)` results in either `1` or `-1` thus flipping the sign of b[idx] swapping a +- to -
        c[idx] = a[idx] + (2*(idx & 1)-1) * b[idx];
    }
}

// main is the main entrypoint.
int main(void)
{
    float *a, *b, *c0, *c1; // host copies of a, b, c
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
    c0 = (float *)malloc(size);
    c1 = (float *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU & copy results (original code)
    vecads_orig<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c0, d_c, size, cudaMemcpyDeviceToHost);

    // regularized
    vecads_reg<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c1, d_c, size, cudaMemcpyDeviceToHost);

    // Verify computation
    bool ok = true;
    for (int idx = 0; idx < N; idx++)
    {
        if (c0[idx] != c1[idx])
        {
            ok = false;
            printf("error: missmatch at index %d, expected: %f got: %f\n", idx, c0[idx], c1[idx]);
        }
    }
    if(!ok) {
        printf("[!] Assertion false (vecads_orig != vecads_reg)\n");
    } else {
        printf("[+] Assertion true (vecads_orig == vecads_reg)\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c0);
    free(c1);
    return 0;
}