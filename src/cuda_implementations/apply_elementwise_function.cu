#include "cuda_matrix_lib.h"
#include "cuda_kernels.h"


__device__ __host__ float elementwise_function(float input) {
    return -1 * input;
}

__device__ pointerToElementwiseFunction_t d_pointer_to_elementwise_function = elementwise_function;



std::vector<float> CudaMatrixLib::test(std::vector<float> input_vector) {    
    
    int input_array_size = input_vector.size();
    int output_array_size = input_array_size;

    int* h_input_array_size = &(input_array_size);
    float* h_input_array = &input_vector[0]; 
    float* h_output_array = (float*) malloc(sizeof(float) * input_array_size);

    int* d_input_array_size;
    float* d_input_array; 
    float* d_output_array;
    
    pointerToElementwiseFunction_t h_pointer_to_elementwise_function;


    const int thread_count = min(input_array_size/ OPERATIONS_PER_THREAD + 1, 1024);
    const int block_count = input_array_size / OPERATIONS_PER_THREAD / 1024 + 1;


    cudaMalloc((void**) &d_input_array, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_output_array, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_input_array_size, sizeof(int));

    cudaMemcpy(d_input_array, h_input_array, sizeof(float) * input_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_array_size, h_input_array_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyFromSymbol(&h_pointer_to_elementwise_function, d_pointer_to_elementwise_function, sizeof(pointerToElementwiseFunction_t), 0, cudaMemcpyDeviceToHost);
    
    auto program_start_time = std::chrono::high_resolution_clock::now();
    
    apply_elementwise_function_kernel <<<block_count, thread_count>>> (d_input_array, h_pointer_to_elementwise_function, d_output_array, d_input_array_size);
    cudaDeviceSynchronize();

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_array, d_output_array, sizeof(float) * output_array_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input_array_size);
    cudaFree(d_input_array);
    cudaFree(d_output_array);  


    std::vector<float> result = c_array_to_std_vector(h_output_array, output_array_size);

    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda scalar multiplication program: " << program_duration << std::endl;

    return result;
}