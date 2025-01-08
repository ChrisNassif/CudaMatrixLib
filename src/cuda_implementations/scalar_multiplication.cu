#include "cuda_matrix_lib.h"
#include "cuda_kernels.h"


std::vector<float> CudaMatrixLib::scalar_multiplication(std::vector<float> input_vector, float input_scalar) {    
    
    int input_array_size = input_vector.size();
    int output_array_size = input_array_size;

    int* h_input_array_size = &(input_array_size);
    float* h_input_array = &input_vector[0]; 
    float* h_input_scalar = &input_scalar;
    float* h_output_array = (float*) malloc(sizeof(float) * input_array_size);

    int* d_input_array_size;
    float* d_input_array; 
    float* d_input_scalar;
    float* d_output_array;


    const int thread_count = min(input_array_size/ OPERATIONS_PER_THREAD + 1, 1024);
    const int block_count = input_array_size / OPERATIONS_PER_THREAD / 1024 + 1;


    cudaMalloc((void**) &d_input_array, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_input_scalar, sizeof(float));
    cudaMalloc((void**) &d_output_array, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_input_array_size, sizeof(int));

    cudaMemcpy(d_input_array, h_input_array, sizeof(float) * input_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_scalar, h_input_scalar, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_array_size, h_input_array_size, sizeof(int), cudaMemcpyHostToDevice);
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    scalar_multiplication_kernel <<<block_count, thread_count>>> (d_input_array, d_input_scalar, d_output_array, d_input_array_size);
    cudaDeviceSynchronize();

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_array, d_output_array, sizeof(float) * output_array_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input_array_size);
    cudaFree(d_input_array);
    cudaFree(d_input_scalar);
    cudaFree(d_output_array);  


    std::vector<float> result = c_array_to_std_vector(h_output_array, output_array_size);

    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda scalar multiplication program: " << program_duration << std::endl;

    return result;
}
