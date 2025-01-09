#include "helper_function_library.hpp"

void print_float_array(float* float_array, int array_size) {
    printf("[");
    int index;
    for (int index = 0; index < array_size; index++) {
        if (index == array_size - 1) {
            printf("%f]\n", float_array[index]);
            continue;
        }
        printf("%f, ", float_array[index]);
    }
}


void print_vector(std::vector<float> vector) {

    printf("[");
    int index;
    for (index = 0; index < vector.size(); index++) {
        if (index == vector.size() - 1) {
            printf("%f]\n", vector[index]);
            continue;
        }
        printf("%f, ", vector[index]);
    }
}


void print_matrix(std::vector<std::vector<float>> matrix) {
    for(int i=0 ; i < matrix.size() ; i++) {
        for(int j=0 ; j < matrix[0].size() ; j++)
            std::cout << matrix[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// void print_matrix(std::vector<std::vector<float>> matrix) {
//     printf("[");
//     int index;
//     for (index = 0; index < vector.size(); index++) {
//         if (index == vector.size() - 1) {
//             printf("%f]\n", vector[index]);
//             continue;
//         }
//         printf("%f, ", vector[index]);
//     }
// }

bool are_vectors_equal(std::vector<float> vector1, std::vector<float> vector2) {

    if (vector1.size() != vector2.size()) {
        return false;
    }

    for (int index = 0; index < vector1.size(); index++) {
        if (vector1[index] != vector2[index]) {
            return false;
        }
    }

    return true;
}


std::vector<std::vector<float>> c_array_to_matrix(float* c_array, int number_of_rows, int number_of_columns) {
    int array_size = number_of_columns * number_of_rows;
    
    std::vector<std::vector<float>> result_matrix;
    for (int i = 0; i < number_of_rows; i++) {
        result_matrix.push_back(std::vector<float>());
        for (int j = 0; j < number_of_columns; j++) {
            result_matrix[i].push_back(c_array[i * number_of_columns + j]);
        }
    }

    return result_matrix;
}

void matrix_to_c_array(std::vector<std::vector<float>> input_matrix, float* c_array) {

    if (input_matrix.size() == 0 || input_matrix[0].size() == 0) {
        return;
    }

    int number_of_rows = input_matrix.size();
    int number_of_columns = input_matrix[0].size();

    for (int i = 0; i < number_of_rows; i++) {
        for (int j = 0; j < number_of_columns; j++) {
            c_array[i*number_of_columns + j] = input_matrix[i][j];
        }
    }
}