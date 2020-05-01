#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA driver and context
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;

// number of elements
int N = 256;

void initCUDA()
{
    char *module_file = (char*) "JCudaVectorAddKernel.ptx";
    char *kernel_name = (char*) "add";
    size_t     totalGlobalMem;

    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        cuDeviceGetCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No devices found that support CUDA\n");
        exit(-1);
    }

    cuDeviceGet(&device, 0);

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Error creating context\n");
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Error loading module: %s\n", module_file);
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Error getting kernel function: %s\n", kernel_name);
        cuCtxDetach(context);
        exit(-1);
    }
}


void allocateDeviceArrays(CUdeviceptr *d_a, CUdeviceptr *d_b, CUdeviceptr *d_c)
{
    cuMemAlloc(d_a, sizeof(int) * N);
    cuMemAlloc(d_b, sizeof(int) * N);
    cuMemAlloc(d_c, sizeof(int) * N);
}

void freeDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
}

void launchKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
    int n = N;
    void *args[4] = {&n, &d_a, &d_b, &d_c };

    cuLaunchKernel(function, N, 1, 1,
                            1, 1, 1, 
                            0, 0, args, 0);
}

int main(int argc, char **argv) {
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    clock_t begin = clock();
    
    int a[N], b[N], c[N];
    CUdeviceptr d_a, d_b, d_c;

    // initialize host arrays
    for (int i = 0; i < N; ++i) {
        a[i] = N - i;
        b[i] = i * i;
    }

    // Initialize the driver and create a context for the first device.
    initCUDA();

    allocateDeviceArrays(&d_a, &d_b, &d_c);

    // copy from host to device
    cuMemcpyHtoD(d_a, a, sizeof(int) * N);
    cuMemcpyHtoD(d_b, b, sizeof(int) * N);

    // run kernel
    printf("Launching kernel...\n");
    launchKernel(d_a, d_b, d_c);
    printf("Kernel finished executing.\n");

    // Test result
    cuMemcpyDtoH(c, d_c, sizeof(int) * N);
    for (int i = 0; i < N; ++i) {
        //printf("%d\n", c[i]);
        if (c[i] != a[i] + b[i])
            printf("Error at index %d: Expected %d, Got %d\n", i, a[i]+b[i], c[i]);
    }
    printf("Passed check\n");

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent: %lf seconds\n", time_spent);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.1f ms \n", time);
    
    freeDeviceMemory(d_a, d_b, d_c);
    cuCtxDetach(context);
    return 0;
}
