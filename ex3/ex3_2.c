#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

// NxN * N -> multiplication
#define N 36000

// Uncomment enable debug mode
//#define DEBUG

#define BENCH_ITERS 5

// Uncomment to enable OMP mode
#define OMP_ENABLED

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
int mat_vec_mul(float *res, float *mat, float *vec)
{
    // Note: this only works for parallel matrices
#ifdef OMP_ENABLED
#pragma omp parallel for num_threads(6)
#endif
    for (uint64_t i = 0; i < N / 2; i++)
    {
        // top row
        int y_top = i; // this happens to indicate the "line-coordinate" for the 0-value lower part.
        // bottom row
        int y_bot = N - 1 - i; // move from bottom

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

    // Matrix malloc
    mat = (float *)malloc(sizeof(float) * N * N);
    if(mat == NULL) {
        printf("Failed to allocate mat\n");
        return -1;
    }

    // vector mallocs
    res = (float *)malloc(sizeof(float) * N);
    if(res == NULL) {
        free(mat);
        printf("Failed to allocate res\n");
        return -1;
    }
    vec = (float *)malloc(sizeof(float) * N);
    if(vec == NULL) {
        free(mat);
        free(vec);
        printf("Failed to allocate vec\n");
        return -1;
    }


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

    // Init vector
    for (uint64_t i = 0; i < N; i++)
        vec[i] = 1.0f;


    // multiplication
    double total_duration = 0.0f;
    for (int i = 0; i < BENCH_ITERS; i++) {
      double start_time = omp_get_wtime();
      mat_vec_mul(res, mat, vec);
      double stop_time = omp_get_wtime();

      double duration = stop_time - start_time;
      total_duration += duration;

      printf("Compute time %lfms\n", duration * 1000);
    }
    printf("Avg compute time %lfms\n", total_duration / BENCH_ITERS * 1000);

#ifdef DEBUG
    print_matrix(mat, N, N);
    print_vec(vec, N);
    print_vec(res, N);
#endif

    // Free resources
    free(vec);
    free(mat);
    free(res);

    return 0;
}
