#include "cpu_matrix_lib.h"
#include <chrono>

std::vector<float> CPUMatrixLib::hadamard_product(std::vector<float> vector1, std::vector<float> vector2) {
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    if (vector1.size() != vector2.size()) {
        return std::vector<float>();
    }

    int vector_size = vector1.size();

    std::vector<float> result_vector(vector_size);

    for (int index = 0; index < vector_size; index++) {
        result_vector[index] = vector1[index] * vector2[index];
    }

    auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cpu hadamard product program: " << program_duration << std::endl;

    return result_vector;
}