#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define N 20000
#define EPSILON 0.01
#define TAU 0.00001

double* create_matrix() {
    double* matrix = (double*) malloc(N * N * sizeof(double));
    return matrix;
} 

double* create_vector() {
    double* vector = (double*) malloc(N * sizeof(double));
    return vector;
}

void initialize_vector_spec(double* b, double value) {
    for (int i = 0; i < N; i++) {
        b[i] = value;
    }
}

double norm(double* vector) {
    double init_norm = 0;
    for (int i = 0; i < N; i++) {
        init_norm += vector[i] * vector[i];
    }

    return sqrt(init_norm);
}

double* sub(double* a, double* b) {
    double* result = create_vector();
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double* matrix_mult(double* matrix, double* vector) {
    double* result = create_vector();
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += matrix[i * N + j] * vector[j];
        }

        result[i] = sum;
    }
    return result;
}

void const_mult(double n, double* vector) {
    for (int i = 0; i < N; i++) {
        vector[i] *= n;
    }
}

bool is_solved(double* A, double* b, double* x) {
    double* result = create_vector();
    result = matrix_mult(A, x);
    result = sub(result, b);

    double result_norm = norm(result);
    double b_norm = norm(b);

    free(result);

    // printf("%f\n", result_norm / b_norm);

    if ((result_norm / b_norm) < EPSILON) {
        return true;
    }

    return false;
}

double* solution(double* A, double* b) {
    double* x = create_vector();
    initialize_vector_spec(x, 0.0);

    while(!is_solved(A, b, x)) {
        double* result = create_vector();
        result = matrix_mult(A, x);
        result = sub(result, b);
        const_mult(TAU, result);

        x = sub(x, result);
        printf("%f\n", x[0]);

        free(result);
    }

    return x;
}

void initialize_matrix(double* A) {
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
    }

    for (int i = 0; i < N; i++) {
        A[i * N + i] = 2.0;
    }
}

void initialize_vector(double* b) {
    for (int i = 0; i < N; i++) {
        b[i] = (double)N + 1;
    }
}

int main() {
    double* A = create_matrix();
    initialize_matrix(A);

    double* b = create_vector();
    initialize_vector(b);


    clock_t begin;
    clock_t end;

    // begin = clock();
    double* x = solution(A, b);
    // end = clock();

    // double total_time = (double)(end - begin) / 1000;

    // printf("%f seconds", total_time);

    free(A);
    free(b);
    free(x);

    return 0;

}

