#include <driver_types.h>
#include <iostream>
#include <cassert>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * @brief Matrix Multiplication Method
 * Cuda accelerated in-place matrix multiplication 
 * 
 * @param n Size of the matrix (width*height)
 * @param a First Matrix (flattened)
 * @param b Second Matrix (flattened)
 * @param c Resulting Matrix (flattend)
*/
__global__ void matmul(int n, float* a, float* b, float* c){
    int column = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row < n && column < n){
        float prod = 0.f;
        for(int i=0; i<n; i++){
            prod += a[row*n + i] * b[i*n + column];
        }
        c[row*n+column] = prod;
    }
}

int main() { 
    int N = 1024; 
    int BLOCK_SIZE = 32;
    float* a = new float[N*N];
    float* b = new float[N*N];
    float* c = new float[N*N];
    for (int i = 0; i<N; i++) {
        for (int j = 0; j<N; j++){
            if (i == j){
                a[i*N + j] = 2;
            }
            b[i*N + j] = i+j;
        }
    }
    float* a_g;
    float* b_g;
    float* c_g;

    cudaMalloc((void**) &a_g, N*N*sizeof(float));
    cudaMalloc((void**) &b_g, N*N*sizeof(float));
    cudaMalloc((void**) &c_g, N*N*sizeof(float));

    cudaMemcpy(a_g, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_g, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    matmul<<<dimGrid, dimBlock>>>(N, a_g, b_g, c_g);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(c, c_g, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            assert(c[i*N+j] == b[i*N+j]*2);
        }
    }
    cudaFree(a_g);
    cudaFree(b_g);
    cudaFree(c_g);
    return 0;
}