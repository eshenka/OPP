#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 100
#define EPSILON 0.0001
#define TAU 0.00001

double* create_matrix() {
    double* matrix = (double*) malloc(N * N * sizeof(double));
    return matrix;
} 

double* create_vector() {
    double* vector = (double*) malloc(N * sizeof(double));
    return vector;
}

void initialize_vector(double* b, double value) {
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

double* sub(double* a, double* b, double* result) {
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}

double* matrix_mult(double* matrix, double* vector, double* result) {
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += matrix[i * N + j] * vector[j];
        }

        result[i] = sum;
    }
}

void const_mult(double n, double* vector) {
    for (int i = 0; i < N; i++) {
        vector[i] *= n;
    }
}

bool is_solved(double* A, double* b, double* x, double* result) {
    matrix_mult(A, x, result);
    sub(result, b, result);

    double result_norm = norm(result);
    double b_norm = norm(b);

    if ((result_norm / b_norm) < EPSILON) {
        return true;
    }

    return false;
}

double* solution(double* A, double* b) {
    double* result = create_vector();
    double* x = create_vector();
    initialize_vector(x, 0.0);

    while(!is_solved(A, b, x, result)) {
        const_mult(TAU, result);

        sub(x, result, x);
    }

    free(result);
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

int main() {
    double* A = create_matrix();
    initialize_matrix(A);

    double* b = create_vector();
    initialize_vector(b, (double)N + 1);

    clock_t begin;
    clock_t end;

    double* x = solution(A, b);
    printf("%f\n", x[0]);

    free(A);
    free(b);
    free(x);

    return 0;
}

