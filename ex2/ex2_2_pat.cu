#include <iostream>

__device__ int proj(unsigned int block_Idx, unsigned int thread_Idx) {
    return (int)(thread_Idx + block_Idx * blockDim.x);
}

__global__ void vecads(float *a, float *b, float *c, int n)
{
    int idx;
    idx = proj(blockIdx.x,threadIdx.x);
    /*if(idx<n){
        if(idx & 1) c[idx] = a[idx] + b[idx];
        else c[idx] = a[idx] - b[idx];
    }*/
    printf("idx: %i blockIdx: %i threadIdx: %i blockdim: %i\n\tcondition val: %i a[idx]: %f b[idx]: %f\n\n", idx, blockIdx.x, threadIdx.x, blockDim.x, (((idx & 1) * 2) - 1), a[idx], b[idx]);
    if(idx<n){
        c[idx] = a[idx] + (((idx & 1) * 2) - 1) * b[idx];
    }
}

// a[i] + b[i] - (a[i] - b[i]) = 2 * b[i]

// a[i] +/- b[i]
// +/- kann man über mehrere Arten erreichen:
// a[i] + -1^(not Bedingung) * b[i]
// a[i] + ((Bedingung * 2) - 1)) * b[i]
// a[i] + (Bedingung - not Bedingung) * b[i]

/*
 * Es führt zu keiner Einschränkgung des Wertebereichs, da im Vergleich zur alten Berechnung
 * folgendes gilt:
 * Der letzte Rechenschritt ist ähnlich wie vor der Änderung: statt a[i] + b[i] oder a[i] - b[i] wird a[i] + b[i] oder a[i] + (-b[i]) berechnet.
 * Diese letzte Rechenschritt schränkt den Wertebereich so ein, dass a[i] + b[i] (bzw. a[i] - b[i]) zu keinem over oder underflow führen dürfen.
 * Solange keiner der Schritte davor den Wertebereich mehr einschränken kann, führt die Veränderung zu keiner weiteren Einscränkung des Wertebereichs.
 * 
 * a[i] + -1^(not Bedingung) * b[i]:
 * Bedingung -> entweder 0 oder 1
 * not Bedingung -> entweder 0 oder 1
 * -1^(not Bedingung) -> entweder 1 oder -1
 * -1^(not Bedingung) * b[i] -> entweder b[i] oder -b[i]
 * -> keine Einschränkung des Wertebereichs
 * 
 * a[i] + ((Bedingung * 2) - 1)) * b[i]:
 * Bedingung -> entweder 0 oder 1
 * (Bedingung * 2) -> entweder 0 oder 2
 * ((Bedingung * 2) - 1) -› entweder -1 oder 1
 * ((Bedingung * 2) - 1) * b[i] -> entweder -b[i] oder b[i]
 * -> keine Einschränkung des Wertebereichs
 * 
 * a[i] + (Bedingung - not Bedingung) * b[i]:
 * Bedingung -> entweder 0 oder 1
 * not Bedingung -> entweder 0 oder 1
 * (Bedingung - not Bedingung) -> entweder -1 oder 1
 * (Bedingung - not Bedingung) * b[i] -> entweder -b[i] oder b[i]
 * -> keine Einschränkung des Wertebereichs
 */
#define N 128

int main(){
    float a[N];
    float b[N];
    float c[N];
    for(int i = 0; i < N; ++i){
        a[i] = i;
        b[i] = 2 * i;
        c[i] = -4793;
    }
    float *device_a, *device_b, *device_c;
    cudaMalloc((void**)&device_a, N * sizeof(float));
    cudaMalloc((void**)&device_b, N * sizeof(float));
    cudaMalloc((void**)&device_c, N * sizeof(float));

    cudaMemcpy(device_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, c, N * sizeof(float), cudaMemcpyHostToDevice);
    vecads<<<4, 32>>>(device_a, device_b, device_c, N);
    cudaMemcpy(c, device_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i){
        std::cout << c[i] << std::endl;
    }
}