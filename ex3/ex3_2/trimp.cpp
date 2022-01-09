#include <iostream>
#include <omp.h>
#define N 36000

// enable or disable the following define clauses if you want printing the matrix/vectors (or not) or multithreading via OpenMP (or not).
//#define PRINT
//#define OMP_ENABLED

typedef double value_t;

void print_matrix(value_t **matrix){
#ifdef PRINT
    std::cout << std::endl;
    for(int x = 0; x < N; ++x){
        for(int y = 0; y < N; ++y){
            std::cout << matrix[x][y] << ' ';
        }
        std::cout << std::endl;
    }
#endif
}

void print_vector(value_t *vector){
#ifdef PRINT
    std::cout << std::endl << '[';
    for(int i = 0; i < N; ++i){
        std::cout << vector[i] << ((i != N - 1) ? ", ": "]");
    }
    std::cout << std::endl;
#endif
}


int main(){
    value_t **matrix;
    value_t *result;
    value_t *vector;
    matrix = (value_t**)malloc(sizeof(value_t*) * N);
    result = (value_t*)malloc(sizeof(value_t) * N);
    vector = (value_t*)malloc(sizeof(value_t) * N);
    for(int i = 0; i < N; ++i){
        matrix[i] = (value_t*)malloc(sizeof(value_t) * N);
    }

    // init
    for(int x = 0; x < N; ++x){
        for(int y = 0; y < x; ++y){
            matrix[x][y] = 0;
        }
        for(int y = x; y < N; ++y){
            matrix[x][y] =  rand();
        }
    }for(int i = 0; i < N; ++i){
        vector[i] = rand();
    }

    // init result with values -1 for easier debugging
    for(int x = 0; x < N; ++x) {
        result[x] = -1;
    }
    print_matrix(matrix);
    print_vector(vector);
    print_vector(result);

    // multiplication
    double start_time = omp_get_wtime();

#ifdef OMP_ENABLED
#pragma omp parallel for
#endif
    for (int thread = 0; thread <= N / 2; ++thread) {
        int index_row_top = thread;
        int index_row_bottom = N - thread - 1;
        value_t *current_row_matrix;
        if (thread == N / 2) {
            current_row_matrix = matrix[thread];
            value_t sum = 0;
            for (int x = N - 1; x >= N / 2; --x) {
                sum += current_row_matrix[x] * vector[x];
            }
            result[thread] = sum;
        } else {
            current_row_matrix = matrix[index_row_top];
            value_t sum = 0;
            for (int i = thread; i < N; ++i) {
                sum += current_row_matrix[i] * vector[i];
            }
            result[index_row_top] = sum;
            sum = 0;
            current_row_matrix = matrix[index_row_bottom];
            for (int i = N - 1 - thread; i < N; ++i) {
                sum += current_row_matrix[i] * vector[i];
            }
            result[index_row_bottom] = sum;
        }
    }
    std::cout << "Compute took " << (omp_get_wtime() - start_time) << "s" << std::endl;
    print_vector(result);
}