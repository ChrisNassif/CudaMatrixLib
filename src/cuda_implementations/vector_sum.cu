#include "cuda_matrix_lib.h"

__global__ void vector_sum_kernel(float* d_vector, float* d_output, int* vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // if (thread_id >= *vector_size) return;

    // d_output_vector[thread_id] = d_vector1[thread_id] + d_vector2[thread_id];

}


float CudaMatrixLib::vector_sum(std::vector<float> vector) {    

    int vector_size = vector.size();
    int* h_vector_size = &vector_size;
    int* d_vector_size;

    float* h_vector = &vector[0];
    float* h_output = (float*) malloc(sizeof(float) * vector_size);

    float* d_vector; float* d_output;


    const int threadCount = min(vector_size, 1024);
    const int blockCount = vector_size / 1024 + 1;


    cudaMalloc((void**) &d_vector, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_output, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector_size, sizeof(int));

    cudaMemcpy(d_vector, h_vector, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_size, h_vector_size, sizeof(int), cudaMemcpyHostToDevice);
    
    vector_sum_kernel <<<blockCount, threadCount>>> (d_vector, d_output, d_vector_size);

    cudaMemcpy(h_output, d_output, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector);       
    cudaFree(d_output);  
    cudaFree(d_vector_size);


    return *h_output;
}
