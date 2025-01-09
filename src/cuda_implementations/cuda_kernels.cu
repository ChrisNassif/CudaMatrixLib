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

typedef float(*pointFunction_t)(float);

__global__ void apply_elementwise_function_kernel(float* d_input_vector, pointFunction_t d_function_to_apply, float* d_output_vector, int* d_input_vector_size) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_start_index = thread_id * OPERATIONS_PER_THREAD;
    
    for (int operation_index = 0; operation_index < OPERATIONS_PER_THREAD; operation_index++) {
        int vector_index = thread_start_index + operation_index;

        if (vector_index >= *d_input_vector_size) return;

        d_output_vector[vector_index] = (*d_function_to_apply)(d_input_vector[vector_index]);
    }
}