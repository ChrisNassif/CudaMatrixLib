cmake_minimum_required(VERSION 3.18)
project(CudaMatrixLib)

# Enable CUDA language support
enable_language(CUDA)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# Set the CUDA architecture
set(CUDA_ARCHITECTURES 75)

# grab all of the cuda files and the cpu files and the utils files
file(GLOB cuda_src_files
    "src/cuda_implementations/**"
)

file(GLOB cpu_src_files
    "src/cpu_implementations/**"
)

file(GLOB utils_files
    "src/utils/**"
)



# Add an executable that uses CUDA
add_executable(CudaMatrixLib 
    src/main.cpp
    ${cuda_src_files}
    ${cpu_src_files}
    ${utils_files}
)

add_executable(unit_tests 
    src/unit_tests.cpp
    ${cuda_src_files}
    ${cpu_src_files}
    ${utils_files}
)


# Set target properties for CUDA
set_target_properties(CudaMatrixLib PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)