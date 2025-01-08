#include "cpu_matrix_lib.h"


std::vector<float> CPUMatrixLib::hadamard_product(std::vector<float> input_vector1, std::vector<float> input_vector2) {
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    if (input_vector1.size() != input_vector2.size()) {
        return std::vector<float>();
    }

    int input_vector_size = input_vector1.size();

    std::vector<float> result_vector(input_vector_size);

    for (int index = 0; index < input_vector_size; index++) {
        result_vector[index] = input_vector1[index] * input_vector2[index];
    }

    auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cpu hadamard product program: " << program_duration << std::endl;

    return result_vector;
}

std::vector<float> CPUMatrixLib::vector_addition(std::vector<float> input_vector1, std::vector<float> input_vector2) {
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    if (input_vector1.size() != input_vector2.size()) {
        return std::vector<float>();
    }

    int input_vector_size = input_vector1.size();

    std::vector<float> result_vector(input_vector_size);

    for (int index = 0; index < input_vector_size; index++) {
        result_vector[index] = input_vector1[index] + input_vector2[index];
    }

    auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cpu vector addition product program: " << program_duration << std::endl;

    return result_vector;
}


float CPUMatrixLib::vector_sum(std::vector<float> input_vector) {
    
    auto program_start_time = std::chrono::high_resolution_clock::now();

    int input_vector_size = input_vector.size();
    double result = 0;

    for (int index = 0; index < input_vector_size; index++) {
        result += input_vector[index];
    }

    auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cpu vector sum product program: " << program_duration << std::endl;

    return result;
}



float CPUMatrixLib::dot_product(std::vector<float> input_vector1, std::vector<float> input_vector2) {
    return CPUMatrixLib::vector_sum(CPUMatrixLib::hadamard_product(input_vector1, input_vector2));
}


std::vector<float> CPUMatrixLib::scalar_multiplication(std::vector<float> input_vector, float input_scalar) {
    auto program_start_time = std::chrono::high_resolution_clock::now();

    int input_vector_size = input_vector.size();
    std::vector<float> result(input_vector_size);

    for (int index = 0; index < input_vector_size; index++) {
        result[index] *= input_scalar;
    }


    auto program_end_time = std::chrono::high_resolution_clock::now();
    float program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end_time - program_start_time).count();
    std::cout << "Time (microseconds) in cpu scalar multiplication program: " << program_duration << std::endl;

    return result;
}