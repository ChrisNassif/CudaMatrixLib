#include <cuda.h>

void move_float_array_from_cpu_to_gpu(float* cpu_data_array, int array_size, float* gpu_data_array) {
    cudaMalloc((void**) &cpu_data_array, sizeof(float) * array_size);
    cudaMemcpy(gpu_data_array, cpu_data_array, sizeof(float) * array_size, cudaMemcpyHostToDevice);
}

void move_float_array_from_gpu_to_cpu(float* gpu_data_array, int array_size, float* cpu_data_array) {
    cudaMemcpy(cpu_data_array, gpu_data_array, sizeof(float) * array_size, cudaMemcpyDeviceToHost);
    cudaFree(gpu_data_array);      
}