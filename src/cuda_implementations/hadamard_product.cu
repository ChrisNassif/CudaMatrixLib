#include <cuda.h>
#include "cuda_matrix_lib.h"
#include <chrono>

#define OPERATIONS_PER_THREAD 1024

__global__ void hadamard_product_kernel(float* d_vector1, float* d_vector2, float* d_output_vector, int* vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;

    for (int index = 0; index < OPERATIONS_PER_THREAD; index++) {
        int input_vector_index = thread_start_index + index;

        if (input_vector_index >= *vector_size) return;

        d_output_vector[input_vector_index] = d_vector1[input_vector_index] * d_vector2[input_vector_index];
    }
}


std::vector<float> CudaMatrixLib::hadamard_product(std::vector<float> vector1, std::vector<float> vector2) {

    int device_count;
    cudaGetDeviceCount(&device_count);

    std::cout << device_count << std::endl;

    // auto program_start_time = std::chrono::high_resolution_clock::now();

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
    

    std::cout << "Cuda Thread Count: " << thread_count << std::endl;
    std::cout << "Cuda Block Count: " << block_count << std::endl;


    cudaMalloc((void**) &d_vector1, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector2, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_output_vector, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector_size, sizeof(int));

    cudaMemcpy(d_vector1, h_vector1, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, h_vector2, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_size, h_vector_size, sizeof(int), cudaMemcpyHostToDevice);
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    hadamard_product_kernel <<<block_count, thread_count>>> (d_vector1, d_vector2, d_output_vector, d_vector_size);
    cudaDeviceSynchronize();

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_vector, d_output_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);      
    cudaFree(d_vector2);        
    cudaFree(d_output_vector);  
    cudaFree(d_vector_size);

    std::vector<float> result;
    result.insert(result.end(), h_output_vector, h_output_vector + vector_size);

    
    // auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda hadamard product program: " << program_duration << std::endl;

    return result;
}

