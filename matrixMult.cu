#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
    
void print_matrix(const float *A, int A_rows, int A_cols) {
  for(int i = 0; i < A_rows; ++i) {
    for(int j = 0; j < A_cols; ++j) {
	  printf("%f ", A[j * A_rows + i]);
	}
	printf("\n");
  }
}

void generate_matrix(float *A, int A_rows, int A_cols) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
  
  curandGenerateUniform(prng, A, A_rows * A_cols);
}

// A(m,k) * B(k,n) = C(m,n)
void cublasMultiply(float *A, float *B, float *C, int m, int k, int n) {
  const float alp = 1.0;
  const float bet = 0.0;
  const float *alpha = &alp;
  const float *beta = &bet;
  
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C, m);
  
  cublasDestroy(handle);
}

void multiplyMatrix(int A_rows, int A_cols, int B_rows, int B_cols) {
  float time;
  cudaEvent_t start, stop;
    
  float* host_matrix_A = (float *) malloc(A_rows * A_cols * sizeof(float));
  float* host_matrix_B = (float *) malloc(B_rows * B_cols * sizeof(float));
  float* host_matrix_C = (float *) malloc(A_rows * B_cols * sizeof(float));
  
  float *device_matrix_A, *device_matrix_B, *device_matrix_C;
  cudaMalloc(&device_matrix_A, A_rows * A_cols * sizeof(float));
  cudaMalloc(&device_matrix_B, B_rows * B_cols * sizeof(float));
  cudaMalloc(&device_matrix_C, A_rows * B_cols * sizeof(float));
  
  generate_matrix(device_matrix_A, A_rows, A_cols);
  cudaMemcpy(host_matrix_A, device_matrix_A, A_rows * A_cols * sizeof(float), cudaMemcpyDeviceToHost);
  //printf("Printing matrix A...\n");
  //print_matrix(host_matrix_A, A_rows, A_cols);
    
  generate_matrix(device_matrix_B, B_rows, B_cols);
  cudaMemcpy(host_matrix_B, device_matrix_B, B_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);
  //printf("\nPrinting matrix B...\n");
  //print_matrix(host_matrix_B, B_rows, B_cols);
  
  // Time operation of interest
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  cublasMultiply(device_matrix_A, device_matrix_B, device_matrix_C, A_rows, A_cols, B_cols);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
    
  cudaMemcpy(host_matrix_C, device_matrix_C, A_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);
  //printf("\nPrinting matrix C...\n");
  //print_matrix(host_matrix_C, A_rows, B_cols);
  
  printf("\nCUBLAS Multiplication execution time:  %3.1f ms\n", time);
  
  free(host_matrix_A);
  free(host_matrix_B);
  free(host_matrix_C);
  cudaFree(device_matrix_A);
  cudaFree(device_matrix_B);
  cudaFree(device_matrix_C);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    printf("There should be 4 arguments: A_row, A_col, B_row, B_col\n");
    printf("Sample run: matrixMult.exe 3 2 2 3");
	return -1;
  }
    
  int A_rows, A_cols, B_rows, B_cols;
  A_rows = atoi(argv[1]);
  A_cols = atoi(argv[2]);
  B_rows = atoi(argv[3]);
  B_cols = atoi(argv[4]);
    
  if (A_cols != B_rows) {
    printf("Columns of matrix 1 dimensions need to match rows matrix 2 dimensions");
    return -1;
  }
  
  printf("\nMultiplying matrix A with dimension %dx%d with matrix B with dimensions %dx%d\n\n", A_rows, A_cols, B_rows, B_cols);
  multiplyMatrix(A_rows, A_cols, B_rows, B_cols);
  
  return 0;
  
}