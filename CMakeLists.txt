cmake_minimum_required(VERSION 3.18)
project(CudaMatrixLib)

# Enable CUDA language support
enable_language(CUDA)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# Set the CUDA architecture
set(CUDA_ARCHITECTURES 75)

# Add an executable that uses CUDA
add_executable(CudaMatrixLib 
    src/main.cpp
    src/hadamard_product.cu
    src/cuda_matrix_lib.cuh
    src/utils/printing_utils.hpp
)

# Set target properties for CUDA
set_target_properties(CudaMatrixLib PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)