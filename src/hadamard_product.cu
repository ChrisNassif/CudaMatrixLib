#include <cuda.h>
#include "cuda_matrix_lib.h"
#include <chrono>


__global__ void hadamard_product_kernel(float* d_vector1, float* d_vector2, float* d_output_vector, int* vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id >= *vector_size) return;

    d_output_vector[thread_id] = d_vector1[thread_id] * d_vector2[thread_id];

}


std::vector<float> hadamard_product(std::vector<float> vector1, std::vector<float> vector2) {

    auto program_start_time = std::chrono::high_resolution_clock::now();

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
    
    hadamard_product_kernel <<<blockCount, threadCount>>> (d_vector1, d_vector2, d_output_vector, d_vector_size);

    cudaMemcpy(h_output_vector, d_output_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);      
    cudaFree(d_vector2);        
    cudaFree(d_output_vector);  
    cudaFree(d_vector_size);

    std::vector<float> result;
    result.insert(result.end(), h_output_vector, h_output_vector + vector_size);

    
    auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda hadamard product program: " << program_duration << std::endl;

    return result;
}

