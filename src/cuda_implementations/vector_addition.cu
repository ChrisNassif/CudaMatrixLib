#include "cuda_matrix_lib.h"

__global__ void vector_addition_kernel(float* d_vector1, float* d_vector2, float* d_output_vector, int* vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id >= *vector_size) return;

    d_output_vector[thread_id] = d_vector1[thread_id] + d_vector2[thread_id];

}


std::vector<float> CudaMatrixLib::vector_addition(std::vector<float> vector1, std::vector<float> vector2) {

    if (vector1.size() != vector2.size()) {
        return std::vector<float>();
    }
    

    int vector_size = vector1.size();
    int* h_vector_size = &vector_size;
    int* d_vector_size;

    float* h_vector1 = &vector1[0]; float* h_vector2 = &vector2[0];
    float* h_output_vector = (float*) malloc(sizeof(float) * vector_size);

    float* d_vector1; float* d_vector2; float* d_output_vector;


    const int threadCount = min(vector_size, 1024);
    const int blockCount = vector_size / 1024 + 1;


    cudaMalloc((void**) &d_vector1, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector2, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_output_vector, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector_size, sizeof(int));

    cudaMemcpy(d_vector1, h_vector1, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, h_vector2, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_size, h_vector_size, sizeof(int), cudaMemcpyHostToDevice);
    
    vector_addition_kernel <<<blockCount, threadCount>>> (d_vector1, d_vector2, d_output_vector, d_vector_size);

    cudaMemcpy(h_output_vector, d_output_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);      
    cudaFree(d_vector2);        
    cudaFree(d_output_vector);  
    cudaFree(d_vector_size);


    std::vector<float> result;
    result.insert(result.end(), h_output_vector, h_output_vector + vector_size); 
    return result;
}
