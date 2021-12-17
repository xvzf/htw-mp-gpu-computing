#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define ROWS 16 // N
#define COLS 10 // N

// Couldn't get it to work, replaced with a something random
/*
__global__ calcmandel(int *a, int N, int M)
{
    int x, y, i, cnt;
    complex z;
    x = proj(blockIdx.x, threadIdx.x);
    y = proj(blockIdx.y, threadIdx.y);
    z = startwert(x, y);
    cnt = 0;
    for (i = 0; i < 255; i++)
        if [(cbetrag(z) < 2)
        {
            z = F(z);
            cnt++;
        }
    a[x][y] = cnt;
}
*/

void fill_random_limited(int *a, int limit, int rowCount, int colsCount)
{
    // Randomly fill buffer as I couldn't get calcmandel to work
    for (int i = 0; i < (rowCount * colsCount); i++)
    {
        a[i] = rand() % limit;
    }
}

// Compute PP[i]=sum(j=0...i-1) num[j] for i=0...N-1, store in num
// Example for N=4: for num={2,4,3,1} the result is num={0,2,6,9}
__device__ void prefix_sum(int *num, int n)
{
    int i, sum, sumold;
    sum = sumold = 0;
    for (i = 0; i < n; i++)
    {
        sum += num[i];
        num[i] = sumold;
        sumold = sum;
    }
}

__global__ void compact(int *a, int *list_x, int *list_y)
{
    __shared__ int num[COLS]; // Shared array

    int idx = threadIdx.x; // Index

    // Count for every row how many entries with 16 exist
    // store in num array (shared state)
    if (idx < COLS)
    {
        int counter = 0;
        // Each thread iterates through all rows at given position
        for (int i = 0; i < ROWS; i++)
        {
            // Check if idx == 16 -> if so, increase counter
            if (a[i * COLS + idx] == 16)
                ;
            {
                counter++;
            }
        }
        // Store final count in shared array
        num[idx] = counter;
    }

    // one of the threads(!) executes after barrier prefix Sum
    __syncthreads();
    if (threadIdx.x == 0)
    {
        prefix_sum(num, COLS);
    }

    // another barrier and store in global liste_x and liste_y (FIXME check if this is what's intended)
    __syncthreads();
    if (idx < COLS)
    {
        int res = num[idx];
        for (int i = 0; i < ROWS; i++)
        {
            if (a[i * COLS + idx] == 16)
            {
                list_x[i * COLS + idx] = i;
                list_y[res] = idx;
                res++;
            }
        }
    }
}

// main is the main entrypoint.
int main(void)
{
    int *mandel, *list_x, *list_y;
    int *d_mandel, *d_list_x, *d_list_y;
    int size = ROWS * COLS * sizeof(int);

    // Allocate space for device
    cudaMalloc((int **)&d_mandel, size);
    cudaMalloc((int **)&d_list_x, ROWS * sizeof(int));
    cudaMalloc((int **)&d_list_y, COLS * sizeof(int));

    // Alloc space for cpu
    mandel = (int *)malloc(size);
    fill_random_limited(mandel, 16, ROWS, COLS);

    list_x = (int *)malloc(ROWS * sizeof(int));
    list_y = (int *)malloc(COLS * sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_mandel, mandel, size, cudaMemcpyHostToDevice);

    // run compact
    compact<<<1, COLS>>>(d_mandel, d_list_x, d_list_y);

    // Copy to host
    cudaMemcpy(list_x, d_list_x, ROWS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(list_y, d_list_y, COLS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ROWS; i++)
    {
        printf("list_x[%d]: %d\n", i, list_x[i]);
    }
    for (int i = 0; i < COLS; i++)
    {
        printf("list_y[%d]: %d\n", i, list_y[i]);
    }

    // Cleanup
    cudaFree(d_mandel);
    cudaFree(d_list_x);
    cudaFree(d_list_y);
    free(mandel);
    free(list_x);
    free(list_y);
    return 0;
}
