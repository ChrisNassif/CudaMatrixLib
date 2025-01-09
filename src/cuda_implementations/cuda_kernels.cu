#include "cuda_kernels.h"


__global__ void hadamard_product_kernel(float* d_input_vector1, float* d_input_vector2, float* d_output_vector, int* d_input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;

    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *d_input_vector_size) return;

        d_output_vector[vector_index] = d_input_vector1[vector_index] * d_input_vector2[vector_index];
    }
}


__global__ void scalar_multiplication_kernel(float* d_input_vector, float* d_input_scalar, float* d_output_vector, int* d_input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;
    
    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *d_input_vector_size) return;

        d_output_vector[vector_index] *= (*d_input_scalar);
    }
}


__global__ void vector_addition_kernel(float* d_input_vector1, float* d_input_vector2, float* d_output_vector, int* d_input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;

    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *d_input_vector_size) return;

        d_output_vector[vector_index] = d_input_vector1[vector_index] + d_input_vector2[vector_index];
    }
}


__global__ void vector_sum_kernel(float* d_input_vector, float* d_output_vector, int* d_input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;
    
    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *d_input_vector_size) return;

        d_output_vector[thread_id] += d_input_vector[vector_index];
    }
}


__global__ void apply_elementwise_function_kernel(float* d_input_vector, pointerToElementwiseFunction_t d_function_to_apply, float* d_output_vector, int* d_input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;
    
    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *d_input_vector_size) return;

        d_output_vector[vector_index] = (*d_function_to_apply)(d_input_vector[vector_index]);
    }
}


__global__ void naive_matrix_multiplication_kernel(
    float* d_input_matrix1, float* d_input_matrix2, float* d_output_matrix, 
    int* d_input_matrix1_number_of_rows_M, int* d_input_matrix1_number_of_columns_K, 
    int* d_input_matrix2_number_of_columns_N
) {
    const int output_matrix_row = threadIdx.y + blockIdx.y * blockDim.y;
    const int output_matrix_column = threadIdx.x + blockIdx.x * blockDim.x;

    const int M = (*d_input_matrix1_number_of_rows_M);
    const int K = (*d_input_matrix1_number_of_columns_K);
    const int N = (*d_input_matrix2_number_of_columns_N);

    if (output_matrix_row >= M || output_matrix_column >= N) return;

    float temp_value = 0;
    // printf("\n");
    for (int i = 0; i < K; i++) {
        // printf("(%i, %i): %i: %i\n", output_matrix_row, output_matrix_column, output_matrix_row*K + i, d_input_matrix1[output_matrix_row*K + i]);
        // printf("(%i, %i): %i: %i\n", output_matrix_row, output_matrix_column, output_matrix_column + i * N, d_input_matrix2[output_matrix_column + i * N]);
        // printf("\n");
        temp_value += d_input_matrix1[output_matrix_row*K + i] * d_input_matrix2[output_matrix_column + i * N];
    }
    // printf("\n");
    // printf("%f\n", temp_value);
    // printf("\n");
    d_output_matrix[output_matrix_row*M + output_matrix_column] = temp_value;

    // const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;
    
    // for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
    //     int vector_index = thread_start_index + operation_index;

    //     int total_number_of_elements
    //     if (vector_index >= (*d_input_matrix1_number_of_rows_M) * (*d_input_matrix2_number_of_columns_N)) return;

    //     d_output_vector[vector_index] = (*d_function_to_apply)(d_input_vector[vector_index]);
    // } 
}