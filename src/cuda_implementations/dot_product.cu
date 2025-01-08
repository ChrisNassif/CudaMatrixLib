#include "cuda_matrix_lib.h"
#include "cuda_kernels.h"

float CudaMatrixLib::dot_product(std::vector<float> input_vector1, std::vector<float> input_vector2) {

    if (input_vector1.size() != input_vector2.size()) {
        return -1;
    }
    return -1;
}