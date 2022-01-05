#include <iostream>
#include <chrono>

#define N 100000000ll
#define RESULT_COUNT 20000
#define K 0

// enable this define clause, if you want Version A, otherwise Version B will be used.
//#define VERSION_A

#ifdef VERSION_A
#define calcIndex(threadIndex, j) (((N) / (RESULT_COUNT)) * (threadIndex) + (j))
#else
#define calcIndex(threadIndex, j) ((threadIndex) + (j) * (RESULT_COUNT))
#endif

// error check with nice line numbers and file name shamelessly stolen from https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


typedef long long index_T;
typedef float value_T;

__global__ void sum(value_T* array, value_T* results){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < RESULT_COUNT){
        value_T sum = 0;
        for(int j = 0; j < (N / RESULT_COUNT); ++j){
            int index = calcIndex(idx, j);
            sum += array[index];
        }
        results[idx] = sum;
    }
}

int main() {
    int threads_per_block = 32 * pow(5, K);
    value_T *array, *results;
    value_T *device_array, *device_results;
    array = (value_T*)malloc(sizeof(value_T) * N);
    results = (value_T*)malloc(sizeof (value_T) * RESULT_COUNT);
    for(index_T i = 0; i < N; ++i){
        array[i] = rand();
    }
    gpuErrchk(cudaMalloc((void**)&device_array, N * sizeof(value_T)));
    gpuErrchk(cudaMalloc((void**)&device_results, RESULT_COUNT * sizeof(value_T)));
    gpuErrchk(cudaMemcpy(device_array, array, N * sizeof(value_T), cudaMemcpyHostToDevice));
    auto start = std::chrono::high_resolution_clock::now();
    sum<<<(RESULT_COUNT + threads_per_block - 1)/threads_per_block, threads_per_block>>>(device_array, device_results);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(results, device_results, sizeof(value_T) * RESULT_COUNT, cudaMemcpyDeviceToHost));

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Computation took "<< duration.count()<< "ms" << std::endl;
    return 0;
}
