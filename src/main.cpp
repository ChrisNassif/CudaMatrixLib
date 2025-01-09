#include "cpu_implementations/cpu_matrix_lib.h"
#include "cuda_implementations/cuda_matrix_lib.h"

int main( void ) {

    int vector_size = 10;
    std::vector<float> vector1(vector_size, 0);
    std::vector<float> vector2(vector_size, 0);

    int index;
    for (index = 0; index < vector_size; index++) {
        vector1[index] = (float)(rand() % 10);
        vector2[index] = (float)(rand() % 10);
    }
    
    // auto cuda_result = CudaMatrixLib::hadamard_product(vector1, vector2);
    // auto cpu_result = CPUMatrixLib::hadamard_product(vector1, vector2);
    // std::cout << "Are cuda and cpu results equal: " << are_vectors_equal(cuda_result, cpu_result) << std::endl;

    // VECTOR SUM DOESNT WORK
    // auto cuda_result = CudaMatrixLib::test(vector1);
    // auto cpu_result = CPUMatrixLib::scalar_multiplication(vector1, 4);
    // std::cout << cuda_result << std::endl;
    // std::cout << cpu_result << std::endl;
    // std::cout << "Are cuda and cpu results equal: " << are_vectors_equal(cuda_result, cpu_result) << std::endl;
    // print_vector(cuda_result);
    Matrix matrix1 = {std::vector<float>({1, 2, 3}), std::vector<float>({3, 4, 5})};
    Matrix matrix2 = {std::vector<float>({1, 2}), std::vector<float>({2, 3}), std::vector<float>({3, 4})};
    
    print_matrix(CudaMatrixLib::naive_matrix_multiplication(matrix1, matrix2));
    return 0;
}
