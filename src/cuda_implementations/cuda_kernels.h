#include <cuda.h>
#include <cstdio>

#define OPERATIONS_PER_THREAD 1024
typedef float(*pointerToElementwiseFunction_t)(float);


__global__ void scalar_multiplication_kernel(float* d_input_vector, float* d_input_scalar, float* d_output_vector, int* d_input_vector_size);
__global__ void hadamard_product_kernel(float* d_input_vector1, float* d_input_vector2, float* d_output_vector, int* d_input_vector_size);
__global__ void vector_sum_kernel(float* d_input_vector, float* d_output_vector, int* d_input_vector_size);
__global__ void vector_addition_kernel(float* d_input_vector1, float* d_input_vector2, float* d_output_vector, int* d_input_vector_size);

__global__ void apply_elementwise_function_kernel(float* d_input_vector, pointerToElementwiseFunction_t d_function_to_apply, float* d_output_vector, int* d_input_vector_size);
__global__ void naive_matrix_multiplication_kernel(
    float* d_input_matrix1, float* d_input_matrix2, float* d_output_matrix, 
    int* d_input_matrix1_number_of_rows_M, int* d_input_matrix1_number_of_columns_K, 
    int* d_input_matrix2_number_of_columns_N
);