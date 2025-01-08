#include "cpu_implementations/cpu_matrix_lib.h"
#include "cuda_implementations/cuda_matrix_lib.h"
#include "utils/printing_utils.hpp"

int main( void ) {

    int vector_size = 10000000;
    std::vector<float> vector1(vector_size, 0);
    std::vector<float> vector2(vector_size, 0);

    int index;
    for (index = 0; index < vector_size; index++) {
        vector1[index] = (float)(rand() % 10);
        vector2[index] = (float)(rand() % 10);
    }
    
    CudaMatrixLib::hadamard_product(vector1, vector2);
    CPUMatrixLib::hadamard_product(vector1, vector2);

    return 0;
}
