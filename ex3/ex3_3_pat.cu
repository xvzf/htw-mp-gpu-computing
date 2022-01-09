#include <iostream>
#include <chrono>

#define N 20000ll
#define threads_per_block 512
typedef double value_T;
typedef long long index_T;


__global__ void mult(value_T *matrix, value_T *vector, value_T *results){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    value_T sum = 0;
    if(idx < N / 2){
        for (int i = idx; i < N; ++i) {
            sum += matrix[N * idx + i] * vector[i];
        }
        results[idx] = sum;
        sum = 0;
        for (int i = N - 1 - idx; i < N; ++i) {
            sum += matrix[N * (N - idx - 1) + i] * vector[i];
        }
        results[N - idx - 1] = sum;
        return;
    }else if(idx == N / 2){
        for (int x = N - 1; x >= N / 2; --x) {
            sum += matrix[N * idx + x] * vector[x];
        }
        results[idx] = sum;
        return;
    }
}


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

int main() {
    value_T *matrix, *vector, *results;
    value_T *device_matrix, *device_vector, *device_results;
    matrix = (value_T*)malloc(sizeof(value_T) * N * N);
    results = (value_T*)malloc(sizeof (value_T) * N);
    vector = (value_T*) malloc(sizeof(value_T) * N);
    for(index_T i = 0; i < N * N; ++i){
        matrix[i] = rand();
    }
    auto start_with_gpumem = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMalloc((void**)&device_matrix, N * N * sizeof(value_T)));
    gpuErrchk(cudaMalloc((void**)&device_results, N * sizeof(value_T)));
    gpuErrchk(cudaMalloc((void**)&device_vector, N * sizeof(value_T)));
    gpuErrchk(cudaMemcpy(device_matrix, matrix, N * N * sizeof(value_T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_vector, vector,  N * sizeof(value_T), cudaMemcpyHostToDevice));
    auto start = std::chrono::high_resolution_clock::now();
    mult<<<(((N + 1) / 2) + threads_per_block - 1)/threads_per_block, threads_per_block>>>(device_matrix, device_vector, device_results);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(results, device_results, sizeof(value_T) * N, cudaMemcpyDeviceToHost));
    auto end_with_gpumem = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_with_gpumem = std::chrono::duration_cast<std::chrono::milliseconds>(end_with_gpumem - start_with_gpumem);
    std::cout << "Computation took "<< duration.count()<< "ms" << std::endl;
    std::cout << "Computation (including memory alloc and copy on gpu) took "<< duration_with_gpumem.count()<< "ms" << std::endl;
    return 0;  // implied cudaFree and free
}
