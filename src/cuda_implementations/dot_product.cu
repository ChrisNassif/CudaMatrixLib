#include "cuda_matrix_lib.h"

float CudaMatrixLib::dot_product(std::vector<float> vector1, std::vector<float> vector2) {
    return CudaMatrixLib::vector_sum(CudaMatrixLib::hadamard_product(vector1, vector2));
}