#ifndef HELPER_FUNCTION_LIBRARY_HPP
#define HELPER_FUNCTION_LIBRARY_HPP

#include <vector>
#include <iostream>

void print_vector(std::vector<float> vector);

bool are_vectors_equal(std::vector<float> vector1, std::vector<float> vector2);

template <typename T> 
std::vector<T> c_array_to_std_vector(T* c_array, int c_array_size);

#include "helper_function_library.tpp"

#endif