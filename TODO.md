matrix multiplication
matrix inverse
matrix determinant

fix accidental block launches by tweaking: const int block_count = input_array_size / OPERATIONS_PER_THREAD / 1024 + 1;
if the input_array_size = 1024 * OPERATIONS_PER_THREAD then we launch 2 blocks, but we don't need to launch 2 blocks