#include "cuda_matrix_lib.h"
#include "cuda_kernels.h"


std::vector<float> CudaMatrixLib::vector_addition(std::vector<float> input_vector1, std::vector<float> input_vector2) {

    if (input_vector1.size() != input_vector2.size()) {
        return std::vector<float>();
    }
    
    int input_array_size = input_vector1.size();
    int output_array_size = input_array_size;

    int* h_input_array_size = &(input_array_size);
    float* h_input_array1 = &input_vector1[0]; 
    float* h_input_array2 = &input_vector2[0];
    float* h_output_array = (float*) malloc(sizeof(float) * input_array_size);

    int* d_input_array_size;
    float* d_input_array1; 
    float* d_input_array2; 
    float* d_output_array;


    const int thread_count = min(input_array_size/ OPERATIONS_PER_THREAD + 1, 1024);
    const int block_count = input_array_size / OPERATIONS_PER_THREAD / 1024 + 1;


    cudaMalloc((void**) &d_input_array1, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_input_array2, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_output_array, sizeof(float) * input_array_size);
    cudaMalloc((void**) &d_input_array_size, sizeof(int));

    cudaMemcpy(d_input_array1, h_input_array1, sizeof(float) * input_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_array2, h_input_array2, sizeof(float) * input_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_array_size, h_input_array_size, sizeof(int), cudaMemcpyHostToDevice);
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    vector_addition_kernel <<<block_count, thread_count>>> (d_input_array1, d_input_array2, d_output_array, d_input_array_size);
    cudaDeviceSynchronize();

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_array, d_output_array, sizeof(float) * output_array_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input_array_size);
    cudaFree(d_input_array1);      
    cudaFree(d_input_array2);        
    cudaFree(d_output_array);  


    std::vector<float> result = c_array_to_std_vector(h_output_array, output_array_size);

    
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda vector addition program: " << program_duration << std::endl;

    return result;
}
