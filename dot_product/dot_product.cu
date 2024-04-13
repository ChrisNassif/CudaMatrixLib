#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>    
#include <vector> 



__global__ void dot_product_kernel(float* d_vector1, float* d_vector2, float* d_output_vector) {
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    d_output_vector[thread_id] = d_vector1[thread_id] * d_vector2[thread_id];
}

std::vector<float> dot_product(std::vector<float> vector1, std::vector<float> vector2) {

    if (vector1.size() != vector2.size() || vector1.size() >= 1024) {
        return std::vector<float>();
    }

    const int vector_size = vector1.size();

    const int threadCount = vector_size;
    const int blockCount = 1;

    float* h_vector1 = &vector1[0]; float* h_vector2 = &vector2[0];
    float* h_output_vector = (float*) malloc(sizeof(float) * vector_size);

    float* d_vector1; float* d_vector2; float* d_output_vector;


    cudaMalloc((void**) &d_vector1, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_vector2, sizeof(float) * vector_size);
    cudaMalloc((void**) &d_output_vector, sizeof(float) * vector_size);

    cudaMemcpy(d_vector1, h_vector1, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, h_vector2, sizeof(float) * vector_size, cudaMemcpyHostToDevice);

    dot_product_kernel <<<blockCount, threadCount>>> (d_vector1, d_vector2, d_output_vector);

    cudaMemcpy(h_vector1, d_vector1, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vector2, d_vector2, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_vector, d_output_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_output_vector);


    std::vector<float> result;
    result.insert(result.end(), h_output_vector, h_output_vector + vector_size); 
    return result;
}

void print_vector(std::vector<float> vector) {

    printf("[");
    int index;
    for (index = 0; index < vector.size(); index++) {
        if (index == vector.size() - 1) {
            printf("%f]\n", vector[index]);
            continue;
        }
        printf("%f, ", vector[index]);
    }
}

int main( void ) {

    // float vector1[] = (float*) malloc(sizeof(float) * 10);
    // float vector2[] = (float*) malloc(sizeof(float) * 10);

    std::vector<float> vector1 = {0, 1, 3, 1, 4, 2};
    std::vector<float> vector2 = {1, 4, 1, 2, 3, 1};

    // int index;
    // for (index = 0; index < 10; index++) {
    //     vector1[index] = rand() % 10;
    //     vector2[index] = rand() % 10;
    // }

    print_vector(dot_product(vector1, vector2));

    return 0;
}
