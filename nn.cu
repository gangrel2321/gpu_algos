#include <driver_types.h>
#include <iostream>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
/**
 * @brief Foward NN Propagation
 * Cuda accelerated in-place NN-propagation 
 * 
 * @param input_t Transposed data input (batch_size X num_feats)
 * @param w Weights (layer_size X num_feats)
 * @param b Bias (layer_size)
 * @param Z output (layer_size X batch_size)
 * @param layer_size Size of the layer 
 * @param num_feats Number of data features
 * @param batch_size Size of the data batch
*/
__global__ void forward(float* input_t, float* w, float* b, float* Z, int layer_size, int num_feats, int batch_size){
    int i = blockIdx.x*blockDim.x + threadIdx.x; // layer_size idx 
    int k = blockIdx.y*blockDim.y + threadIdx.y; // batch_size idx

    if (k < batch_size && i < num_feats) {
        float cur_val = 0.f;
        for(int j = 0; j < num_feats; j++) { 
            cur_val += w[i*num_feats + j]*input_t[j*num_feats + k];
        }
        Z[i*num_feats + k] = cur_val + b[i];
    }
}


// ------------ Activations ------------

__global__ void relu(int w, int h, float* input, float* output){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y; 
    if (row < h && col < w){
        float activation = input[row*w + col];
        output[row*w + col] = activation > 0.f ? activation : 0.f; 
    }
}

__global__ void relu_backwards(int w, int h, float* a, float* d_l, float* b)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float activation = a[row*w+column];
    b[row*w+column] = activation > 0.f ? d_l[row*w+column] : 0.f;
  }
}

__global__ void softmax(int w, int h, float* input, float* output){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y; 
    if (row < h && col < w){
        float maxval = input[row*w];
        for(int i=1; i<w; i++){
            maxval = max(maxval, input[row*w + i]);
        }
        float divisor = 0.f;
        for(int i = 0; i<w; i++){
            divisor += exp(input[row*w + i] - maxval);
        }
        output[row*w + col] = exp(input[row*w + col] - maxval)/(divisor);
    }
}

// ------------- Losses ---------------

/**
 * @brief Cross Entropy Loss
 * Cuda accelerated cross entropy loss
 * 
 * @param preds Predicted data probs (n x m) 
 * @param data True data labels (n x m)
 * @param output Loss output 
 * @param n height of input matrices
 * @param m width of input matrices
*/
__global__ void cross_entropy(float* preds, float* data, float* output, int n, int m){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        float loss = 0.f;
        for(int j = 0; j<m; j++){
            loss -= data[i*m + j] * log(max(1e-7, preds[i*m + j]));
        }
        output[i] = loss;
    }
}

__global__ void cross_entropy_backwards(float* preds, float* data, float* output, int n, int m){
    int i = blockIdx.x*blockDim.x + threadIdx.x; 
    int j = blockIdx.y*blockDim.y + threadIdx.y; 
    if (i < n && j < m){
        output[i*m + j] = preds[i*m + j] - data[i*m + j];
    }
}

// -------- Layer Initialization -----------


__global__ void init_rand(int w, int h, float* mat)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    curandState state;
    curand_init(42, row*w+column, 0, &state);
    mat[row*w + column] = curand_normal(&state)*sqrtf(2.f/h);
  }
}