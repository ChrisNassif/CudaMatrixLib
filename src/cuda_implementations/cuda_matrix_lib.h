#ifndef CUDA_MATRIX_LIB_CUH
#define CUDA_MATRIX_LIB_CUH

#include <vector> 
#include <iostream>

namespace CudaMatrixLib {
    std::vector<float> hadamard_product(std::vector<float> vector1, std::vector<float> vector2);
    std::vector<float> vector_addition(std::vector<float> vector1, std::vector<float> vector2);
}

#endif