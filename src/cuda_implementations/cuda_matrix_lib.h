#ifndef CUDA_MATRIX_LIB_H
#define CUDA_MATRIX_LIB_H

#include <vector> 
#include <iostream>
#include <chrono>

namespace CudaMatrixLib {
    std::vector<float> vector_addition(std::vector<float> vector1, std::vector<float> vector2);
    std::vector<float> hadamard_product(std::vector<float> vector1, std::vector<float> vector2);
    float vector_sum(std::vector<float> vector);
    float dot_product(std::vector<float> vector1, std::vector<float> vector2);

}

#endif