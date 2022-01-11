#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
// #include <omp.h>

// NxN * N -> multiplication
#define N 20000

// How many benchmark iterations to run
#define BENCH_ITERS 5

// Uncomment enable debug mode
#define DEBUG

#define THREADS_PER_BLOCK 512

void print_matrix(float *a, uint64_t m, uint64_t n)
{

    printf("(\n");
    for (uint64_t i = 0; i < m; i++)
    {
        printf("\t");
        for (uint64_t j = 0; j < n; j++)
        {
            int idx = i * m + j;
            printf("[m=%ld,n=%ld] %0.0f\t", i, j, a[idx]);
        }
        printf("\n");
    }
    printf(")\n");
}

void print_vec(float *a, uint64_t n)
{

    printf("(\n");
    for (uint64_t i = 0; i < n; i++)
    {
        printf("\t[idx=%ld] %0.0f\t", i, a[i]);
        printf("\n");
    }
    printf(")\n");
}

// mat_vec_mul multiplies a (sparse) matrix and a vector and stores the result in res
// the idea is to have each iteration move in from top to bottom row down to the middle.
// This way each iteration handles the same amount of entries
// Note: this only works with even N!!
__global__ void mat_vec_mul(float *res, float *mat, float *vec)
{
    uint64_t idx = threadIdx.x;

    if (idx < (N / 2))
    {
        // top row
        int y_top = idx;
        // bottom row
        int y_bot = N - 1 - idx; // move from bottom

        // Calc top row
        float sum_top = 0.0f;
        for (uint64_t x = y_top; x < N; x++)
            sum_top += mat[y_top * N + x] * vec[x];
        res[y_top] = sum_top;

        // Calc bottom row
        float sum_bot = 0.0f;
        for (uint64_t x = y_bot; x < N; x++)
            sum_bot += mat[y_bot * N + x] * vec[x];
        res[y_bot] = sum_bot;
    }
}

int main()
{
    float *res, *vec, *mat;
    float *d_res, *d_vec, *d_mat;
    struct timespec t0, t1;

    // Matrix malloc
    mat = (float *)malloc(sizeof(float) * N * N);
    cudaMalloc((void **)&d_mat, sizeof(float) * N * N);

    // vector mallocs
    res = (float *)malloc(sizeof(float) * N);
    cudaMalloc((void **)&d_res, sizeof(float) * N);

    vec = (float *)malloc(sizeof(float) * N);
    cudaMalloc((void **)&d_vec, sizeof(float) * N);

    // Init matrix
    for (uint64_t y = 0; y < N; y++)
    {
        // Upper half of matrix
        for (uint64_t x = y; x < N; x++)
            mat[y * N + x] = 1.0f;

        // Lower half of matrix (0)
        for (uint64_t x = 0; x < y; x++)
            mat[y * N + x] = 0.0f;
    }
    cudaMemcpy(d_mat, mat, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Init vector
    for (uint64_t i = 0; i < N; i++)
        vec[i] = 1.0f;
    cudaMemcpy(d_vec, vec, N * sizeof(float), cudaMemcpyHostToDevice);

    // multiplication
    for (int i = 0; i < BENCH_ITERS; i++)
    {
        timespec_get(&t0, TIME_UTC);
        mat_vec_mul<<<(((N + 1) / 2) + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_res, d_mat, d_vec);
        cudaDeviceSynchronize();
        timespec_get(&t1, TIME_UTC);

        cudaMemcpy(res, d_res, N * sizeof(float), cudaMemcpyDeviceToHost);

        double duration = (double)(t1.tv_sec - t0.tv_sec) + ((double)(t1.tv_nsec - t0.tv_nsec)/1000000000L);
        printf("Compute took %lfms\n", duration * 1000);
    }

#ifdef DEBUG
    print_matrix(mat, N, N);
    print_vec(vec, N);
    print_vec(res, N);
#endif

    // Free resources
    free(vec);
    free(mat);
    free(res);
    cudaFree((void **)d_vec);
    cudaFree((void **)d_mat);
    cudaFree((void **)d_res);

    return 0;
}
