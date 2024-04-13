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
