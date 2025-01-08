cmake_minimum_required(VERSION 3.18)
project(CudaMatrixLib)

# Enable CUDA language support
enable_language(CUDA)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# Set the CUDA architecture
set(CUDA_ARCHITECTURES 75)

# grab all of the cuda files
file(GLOB cuda_src_files
    "src/*.cu"
)



# Add an executable that uses CUDA
add_executable(CudaMatrixLib 
    src/main.cpp
    ${cuda_src_files}
    src/utils/printing_utils.hpp
)

# Add an executable that uses CUDA
add_executable(CPUMatrixLib 
    cpu_implementations/main.cpp
    cpu_implementations/hadamard_product.cpp
    cpu_implementations/cpu_matrix_lib.h
)

# Set target properties for CUDA
set_target_properties(CudaMatrixLib PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)