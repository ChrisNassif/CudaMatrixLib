#include <vector>
#include <iostream>

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