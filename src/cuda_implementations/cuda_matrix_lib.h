#ifndef CUDA_MATRIX_LIB_H
#define CUDA_MATRIX_LIB_H
 
#include <iostream>
#include <chrono>
#include "../utils/helper_function_library.hpp"

typedef std::vector<std::vector<float>> Matrix;

namespace CudaMatrixLib {
    std::vector<float> vector_addition(std::vector<float> input_vector1, std::vector<float> input_vector2);
    std::vector<float> hadamard_product(std::vector<float> input_vector1, std::vector<float> input_vector2);
    float vector_sum(std::vector<float> input_vector);
    float dot_product(std::vector<float> input_vector1, std::vector<float> input_vector2);
    std::vector<float> scalar_multiplication(std::vector<float> input_vector, float input_scalar);
    std::vector<float> test(std::vector<float> input_vector);
    Matrix naive_matrix_multiplication(Matrix input_matrix1, Matrix input_matrix2);
}

#endif