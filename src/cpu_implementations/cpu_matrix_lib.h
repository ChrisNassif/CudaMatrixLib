#ifndef CPU_MATRIX_LIB_CUH
#define CPU_MATRIX_LIB_CUH

#include <vector> 
#include <iostream>

namespace CPUMatrixLib {
    std::vector<float> hadamard_product(std::vector<float> vector1, std::vector<float> vector2);
    std::vector<float> vector_addition(std::vector<float> vector1, std::vector<float> vector2);
    float vector_sum(std::vector<float> vector);
}

#endif