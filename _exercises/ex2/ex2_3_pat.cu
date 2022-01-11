__device__ void PrefixSum(int *num,int N){
    int i,sum,sumold; sum=sumold=0;
    for(i=0;i<N;i++){ sum+=num[i]; num[i]=sumold; sumold=sum; }
}

#define N 3
#define M 3
__global__ void compact(int *a, int *listex, int *listey){
    __shared__ int num[N];
    int count = 0;
    const unsigned int idx = threadIdx.x;
    if(idx < N) {  // executed by N threads
        for (int i = M * idx; i < (M * (idx + 1)); ++i) {
            count += a[i] == 16;
        }
        num[idx] = count;
    }
    __syncthreads();
    if(idx == 0){  // executed by exactly one thread with idx == 0
        PrefixSum(num, N);
    }
    __syncthreads();
    if(idx < N) {  // executed by N threads
        int index = num[idx];
        for (int x = 0; x < M; ++x) {
            if (a[idx * M + x] == 16) {
                listex[index] = x;
                listey[index] = idx;
                ++index;
            }
        }
    }
}