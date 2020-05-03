extern "C"

__global__ void add(int n, float *a, float *b, float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
