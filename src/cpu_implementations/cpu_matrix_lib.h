#ifndef CPU_MATRIX_LIB_H
#define CPU_MATRIX_LIB_H

#include <vector> 
#include <iostream>
#include <chrono>

namespace CPUMatrixLib {
    std::vector<float> vector_addition(std::vector<float> input_vector1, std::vector<float> input_vector2);
    std::vector<float> hadamard_product(std::vector<float> input_vector1, std::vector<float> input_vector2);
    float vector_sum(std::vector<float> input_vector);
    float dot_product(std::vector<float> input_vector1, std::vector<float> input_vector2);
    std::vector<float> scalar_multiplication(std::vector<float> input_vector, float input_scalar);
}

#endif