#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#define N 100000000ll
#define RES_COUNT 20000

#define BENCH_ITERATIONS 5
#define BENCH_MAX_K 20

// uncomment to enable index calculation of b
#define B

// random_floats fills an array of floats; not that random :-)
void random_floats(float *a, uint64_t array_size)
{
    for (uint64_t i = 0; i < array_size; i++)
    {
        a[i] = (float)i;
    }
}

__device__ int calc_idx_a(int threadIdx, int j)
{
    return (N / RES_COUNT) * threadIdx + j;
}

__device__ int calc_idx_b(int threadIdx, int j)
{
    return threadIdx + j * RES_COUNT;
}

__global__ void sum(float *in, float *out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Make sure we're not overflowing the target buffer
    if (idx < RES_COUNT)
    {
        float sum = 0;
        // Sum up
        for (int j = 0; j < (N / RES_COUNT); j++)
        {
#ifndef B
            int in_idx = calc_idx_a(idx, j);
#else
            int in_idx = calc_idx_b(idx, j);
#endif
            sum += in[in_idx];
        }
        out[idx] = sum;
    }
}


double run(int k) {
    int num_threads = 32 * pow(5, k);
    float *in, *out, *d_in, *d_out;

    in = (float *)malloc(sizeof(float) * N);
    random_floats(in, N); // Random init
    cudaMalloc((void **)&d_in, sizeof(float) * N);
    cudaMemcpy(d_in, in, N * sizeof(float), cudaMemcpyHostToDevice);

    out = (float *)malloc(sizeof(float) * RES_COUNT);
    cudaMalloc((void **)&d_out, sizeof(float) * RES_COUNT);

    // Start time measurement
    double start_time = omp_get_wtime();

    // Run computation
    sum<<<((RES_COUNT + num_threads) / num_threads), num_threads>>>(d_in, d_out);
    cudaDeviceSynchronize();

    double duration = omp_get_wtime() - start_time;

    // Copy results
    cudaMemcpy(out, d_out, sizeof(float) * RES_COUNT, cudaMemcpyDeviceToHost);

    free(in);
    free(out);
    cudaFree((void**)d_in);
    cudaFree((void**)d_out);

    return duration * 1000;
}

void bench(int k) {
    double total = 0.0f;
    printf("| %d | ", k);
    for(int i = 0; i < BENCH_ITERATIONS; i++) {
        double duration_ms = run(k);
        total += duration_ms;
        printf(" %0.3lfms |", duration_ms);
    }
    printf(" %0.3lfms |\n", total / BENCH_ITERATIONS);
}

int main(void)
{
    // Table Header
    printf("| k |");
    for(int i = 0; i < BENCH_ITERATIONS; i++) printf(" #%d |", i);
    printf(" avg |\n|");
    for(int i = 0; i < BENCH_ITERATIONS+2; i++) printf("---|");
    printf("\n");

    // Iterations
    for(int k = 0; k < BENCH_MAX_K; k++) {
        bench(k);
    }
    return 0;
}
