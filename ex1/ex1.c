/*      Exercise 1 GPU Computing, HTW Saar
 *
 * Author: Matthias Riegler <mriegler@htwsaar.de>
 * 
 * Notes: This does not contain any error handling, use with caution
 * 
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>


// helper from https://gist.github.com/dgoguerra/7194777
static const char *bytes_to_string(uint64_t bytes) {
	char *suffix[] = {"B", "KB", "MB", "GB", "TB"};
	char length = sizeof(suffix) / sizeof(suffix[0]);

	uint64_t i = 0;
	double dblBytes = bytes;

	if (bytes > 1024) {
		for (i = 0; (bytes / 1024) > 0 && i<length-1; i++, bytes /= 1024)
			dblBytes = bytes / 1024.0;
	}

	static char output[200];
	sprintf(output, "%.02lf %s", dblBytes, suffix[i]);
	return output;
}

// initialises a float array of size n with random values
float* init_vector(uint64_t n) {
    float* m = (float *) malloc(n * sizeof(float));
    if(m == NULL) {
        return NULL;
    }
    #pragma omp for
    for(uint64_t i = 0; i < n; i++) {
        // Generate random values from 0-1000
        m[i] = (float)rand()/(float)(RAND_MAX) * 1000.0;
    }
    return m;
}

float* init_matrix(uint64_t n) {
    return init_vector(n*n);
}

// Multiplication of nxn matrix A with vector b
float* mmult(uint64_t n, float* a, float* b, float *c) {
    float sum;

    // Perform matrix multiplications
    for(uint64_t i = 0; i < n; i++) {
        sum = 0.0;
        for(uint64_t j = 0; j < n; j++) {
            sum += a[i*n + j] * b[j];
        }
        c[i] = sum;
    }

    return c;
}

// openmp accelerated Multiplication of nxn matrix A with vector b
float* mmult_openmp(uint64_t n, float* a, float* b, float *c) {

    // Perform matrix multiplications
    // num_threads(4) -> Limit number of threads so MacOS tagets the high performance cores of the M1 processor
    #pragma omp parallel for num_threads(4)
    for(uint64_t i = 0; i < n; i++) {
        float sum = 0.0;
        for(uint64_t j = 0; j < n; j++) {
            sum += a[i*n + j] * b[j];
        }
        c[i] = sum;
    }

    return c;
}

// openmp accelerated Multiplication of nxn matrix A with vector b, non deterministic runtime
float* mmult_openmp_non_deterministic(uint64_t n, float* a, float* b, float *c) {

    // Perform matrix multiplications
    // num_threads(4) -> Limit number of threads so MacOS tagets the high performance cores of the M1 processor
    #pragma omp parallel for num_threads(4) schedule(dynamic, 1)
    for(uint64_t i = 0; i < n; i++) {
        float sum = 0.0;
        for(uint64_t j = 0; j < n; j++) {
            float val = a[i*n + j] * b[j];
            sum += val;
            // count based on multiplication value
            int cntr = 0;
            int limit = (int)val;
            while(cntr < limit) cntr++;
        }
        c[i] = sum;
    }

    return c;
}


int main(int argc, char *argv[]) {
    uint64_t n;
    float *a, *b, *c;
    double begin, end;

    if(argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi(argv[1]);

    // Estimate memory size for matrices
    printf("[+] estimated memory bytes for input/result storage: %s\n", bytes_to_string((n+2)* n * sizeof(float)));

    // Initialise the random nxn matrix
    printf("[+] Aloocate memory and generate random matrix (%llux%llu) and vector (%llu)\n", n, n, n);
    a = init_matrix(n); // Random matrix
    b = init_vector(n); // random vector for multiplication
    c = malloc(n * sizeof(float));
    if(a == NULL || b == NULL || c == NULL) {
        printf("[!] Memory allocation failed, exiting\n");
        return EXIT_FAILURE;
    }
    printf("[+] Memory allocation successful\n");


    // Do computation (normal)
    printf("[+] Starting normal execution\n");
    begin = omp_get_wtime(); // Start time
    c = mmult(n, a, b, c); // Calculation
    end = omp_get_wtime(); // Stop time

    printf("[+] Normal execution time: %0.2f ms\n", (end-begin)*1000);


    // Do computation (openmp)
    printf("[+] Starting openmp accelerated execution\n");
    begin = omp_get_wtime(); // Start time
    c = mmult_openmp(n, a, b, c); // Calculation
    end = omp_get_wtime(); // Stop time
    printf("[+] openmp execution time: %0.2f ms\n", (end-begin)*1000);

    // Do computation (openmp) nondeterministic
    printf("[+] Starting openmp nondeterministic execution\n");
    begin = omp_get_wtime(); // Start time
    c = mmult_openmp_non_deterministic(n, a, b, c); // Calculation
    end = omp_get_wtime(); // Stop time

    printf("[+] openmp nondeterministic execution time: %0.2f ms\n", (end-begin)*1000);

    // Cleanup memory
    free(a);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}