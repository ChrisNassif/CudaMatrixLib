#ifndef HELPER_FUNCTION_LIBRARY_TPP
#define HELPER_FUNCTION_LIBRARY_TPP

#include "helper_function_library.hpp"

template <typename T> 
std::vector<T> c_array_to_std_vector(T* c_array, int c_array_size) {
    std::vector<float> result;
    result.insert(result.end(), c_array, c_array + c_array_size); 
    return result;
}

#endif