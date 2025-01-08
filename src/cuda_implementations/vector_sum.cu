#include "cuda_matrix_lib.h"
// #include "../utils/helper_function_library.hpp"

#define OPERATIONS_PER_THREAD 1024


__global__ void vector_sum_kernel(float* d_vector, float* d_output, int* input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;
    
    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *input_vector_size) return;

        d_output[thread_id] += d_vector[vector_index];
    }

}


float CudaMatrixLib::vector_sum(std::vector<float> vector) {    

    int vector_size = vector.size();
    int* h_vector_size = &vector_size;
    int* d_vector_size;

    float* h_vector = &vector[0];
    int output_vector_size = vector_size/ OPERATIONS_PER_THREAD + 1;
    float* h_output_vector = (float*) malloc(sizeof(float) * output_vector_size);

    float* d_vector; 
    float* d_output_vector;

    const int thread_count = min(vector_size/ OPERATIONS_PER_THREAD + 1, 1024);
    const int block_count = vector_size / OPERATIONS_PER_THREAD / 1024 + 1;

    std::cout << thread_count << std::endl;
    std::cout << block_count << std::endl;

    cudaMalloc((void**) &d_vector, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_output_vector, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector_size, sizeof(int));

    cudaMemcpy(d_vector, h_vector, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_size, h_vector_size, sizeof(int), cudaMemcpyHostToDevice);
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    std::cout << "hi1" << std::endl;
    vector_sum_kernel <<<block_count, thread_count>>> (d_vector, d_output_vector, d_vector_size);
    cudaDeviceSynchronize();
    std::cout << "hi2" << std::endl;

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_vector, d_output_vector, sizeof(float) * output_vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector);       
    cudaFree(d_output_vector);  
    cudaFree(d_vector_size);

    std::vector<float> result1;
    result1.insert(result1.end(), h_output_vector, h_output_vector + output_vector_size); 
    // print_vector(result1);


    // sum up the result
    float result = 0;
    for (int i = 0; i < output_vector_size; i++) {
        result += h_output_vector[i];
    }


    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda vector sum program: " << program_duration << std::endl;

    return result;
}
