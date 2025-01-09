#ifndef HELPER_FUNCTION_LIBRARY_HPP
#define HELPER_FUNCTION_LIBRARY_HPP

#include <vector>
#include <iostream>

void print_float_array(float* float_array, int array_size);
void print_vector(std::vector<float> vector);
void print_matrix(std::vector<std::vector<float>> matrix);

bool are_vectors_equal(std::vector<float> vector1, std::vector<float> vector2);

void matrix_to_c_array(std::vector<std::vector<float>> input_matrix, float* c_array);
std::vector<std::vector<float>> c_array_to_matrix(float* c_array, int number_of_rows, int number_of_columns);

template <typename T> 
std::vector<T> c_array_to_std_vector(T* c_array, int c_array_size);

#include "helper_function_library.tpp"

#endif