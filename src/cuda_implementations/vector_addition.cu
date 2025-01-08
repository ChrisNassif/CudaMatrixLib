#include "cuda_matrix_lib.h"

#define OPERATIONS_PER_THREAD 1024


__global__ void vector_addition_kernel(float* d_vector1, float* d_vector2, float* d_output_vector, int* vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;

    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *vector_size) return;

        d_output_vector[vector_index] = d_vector1[vector_index] + d_vector2[vector_index];
    }

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


    const int thread_count = min(vector_size/ OPERATIONS_PER_THREAD + 1, 1024);
    const int block_count = vector_size / OPERATIONS_PER_THREAD / 1024 + 1;


    cudaMalloc((void**) &d_vector1, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector2, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_output_vector, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector_size, sizeof(int));

    cudaMemcpy(d_vector1, h_vector1, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, h_vector2, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_size, h_vector_size, sizeof(int), cudaMemcpyHostToDevice);
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    vector_addition_kernel <<<block_count, thread_count>>> (d_vector1, d_vector2, d_output_vector, d_vector_size);
    cudaDeviceSynchronize();

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_vector, d_output_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);      
    cudaFree(d_vector2);        
    cudaFree(d_output_vector);  
    cudaFree(d_vector_size);


    std::vector<float> result;
    result.insert(result.end(), h_output_vector, h_output_vector + vector_size); 

    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda vector addition program: " << program_duration << std::endl;

    return result;
}
