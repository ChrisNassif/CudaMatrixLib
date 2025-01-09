#include "cuda_matrix_lib.h"
#include "cuda_kernels.h"


Matrix CudaMatrixLib::naive_matrix_multiplication(Matrix input_matrix1, Matrix input_matrix2) {    
    if (input_matrix1.size() == 0 || input_matrix2.size() == 0) {
        return Matrix();
    }

    int input_matrix1_number_of_rows = input_matrix1.size();
    int input_matrix1_number_of_columns = input_matrix1[0].size();

    int input_matrix2_number_of_rows = input_matrix2.size();
    int input_matrix2_number_of_columns = input_matrix2[0].size();

    if (input_matrix1_number_of_columns != input_matrix2_number_of_rows) {
        return Matrix();
    }

    int output_matrix_number_of_rows = input_matrix1_number_of_rows;
    int output_matrix_number_of_columns = input_matrix2_number_of_columns;
    
    float* h_input_matrix1 = (float*) malloc(sizeof(float) * input_matrix1_number_of_rows * input_matrix1_number_of_columns);
    float* h_input_matrix2 = (float*) malloc(sizeof(float) * input_matrix2_number_of_rows * input_matrix2_number_of_columns);
    float* h_output_matrix = (float*) malloc(sizeof(float) * output_matrix_number_of_rows * output_matrix_number_of_columns);
    int* h_input_matrix1_number_of_rows_M = &(input_matrix1_number_of_rows);
    int* h_input_matrix1_number_of_columns_K = &(input_matrix1_number_of_columns);
    int* h_input_matrix2_number_of_columns_N = &(input_matrix2_number_of_columns);

    matrix_to_c_array(input_matrix1, h_input_matrix1);
    matrix_to_c_array(input_matrix2, h_input_matrix2);

    // print_float_array(h_input_c_matrix1, input_matrix1_number_of_rows * input_matrix1_number_of_columns);
    // print_float_array(h_input_c_matrix2, input_matrix2_number_of_rows * input_matrix2_number_of_columns);
    
    float* d_input_matrix1; 
    float* d_input_matrix2;
    float* d_output_matrix;
    int* d_input_matrix1_number_of_rows_M;
    int* d_input_matrix1_number_of_columns_K;
    int* d_input_matrix2_number_of_columns_N;


    cudaMalloc((void**) &d_input_matrix1, sizeof(float) * input_matrix1_number_of_rows * input_matrix1_number_of_columns);
    cudaMalloc((void**) &d_input_matrix2, sizeof(float) * input_matrix2_number_of_rows * input_matrix2_number_of_columns);
    cudaMalloc((void**) &d_output_matrix, sizeof(float) * output_matrix_number_of_rows * output_matrix_number_of_columns);
    cudaMalloc((void**) &d_input_matrix1_number_of_rows_M, sizeof(int));
    cudaMalloc((void**) &d_input_matrix1_number_of_columns_K, sizeof(int));
    cudaMalloc((void**) &d_input_matrix2_number_of_columns_N, sizeof(int));

    cudaMemcpy(d_input_matrix1, h_input_matrix1, sizeof(float) * input_matrix1_number_of_rows * input_matrix1_number_of_columns, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_matrix2, h_input_matrix2, sizeof(float) * input_matrix2_number_of_rows * input_matrix2_number_of_columns, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_matrix, h_output_matrix, sizeof(float) * output_matrix_number_of_rows * output_matrix_number_of_columns, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_matrix1_number_of_rows_M, h_input_matrix1_number_of_rows_M, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_matrix1_number_of_columns_K, h_input_matrix1_number_of_columns_K, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_matrix2_number_of_columns_N, h_input_matrix2_number_of_columns_N, sizeof(int), cudaMemcpyHostToDevice);

    // TODO: CHANGE THIS
    const auto thread_count = dim3(16, 16);
    const auto block_count = 1;

    auto program_start_time = std::chrono::high_resolution_clock::now();

    naive_matrix_multiplication_kernel <<<block_count, thread_count>>> (
        d_input_matrix1, d_input_matrix2, d_output_matrix,
        d_input_matrix1_number_of_rows_M, d_input_matrix1_number_of_columns_K, d_input_matrix2_number_of_columns_N
    );
    cudaDeviceSynchronize();

    auto program_end_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_matrix, d_output_matrix, sizeof(float) * output_matrix_number_of_rows * output_matrix_number_of_columns, cudaMemcpyDeviceToHost);

    cudaFree(d_input_matrix1);
    cudaFree(d_input_matrix2);
    cudaFree(d_output_matrix);
    cudaFree(d_input_matrix1_number_of_rows_M);  
    cudaFree(d_input_matrix1_number_of_columns_K);
    cudaFree(d_input_matrix2_number_of_columns_N);

    free(h_input_matrix1);
    free(h_input_matrix2);

    print_float_array(h_output_matrix, input_matrix1_number_of_rows * input_matrix2_number_of_columns);

    Matrix result = c_array_to_matrix(h_output_matrix, output_matrix_number_of_rows, output_matrix_number_of_columns);

    free(h_output_matrix);
    
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cuda scalar multiplication program: " << program_duration << std::endl;

    return result;
}
