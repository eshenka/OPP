#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#define N 15000
#define EPSILON 0.0001
#define TAU 0.00001

double* create_matrix(int m) {
    double* matrix = (double*) malloc(m * N * sizeof(double));
    return matrix;
}

double* create_vector(int size) {
    double* vector = (double*) malloc(size * sizeof(double));
    return vector;
}

void initialize_matrix(double* matrix, int m, int shift) {
    for (int i = 0; i < N * m; i++) {
        matrix[i] = 1.0;
    }

    for (int i = 0; i < m; i++) {
        matrix[i * N + i + shift] = 2.0;
    }
}

void initialize_vector(double* vector) {
    for (int i = 0; i < N; i++) {
        vector[i] = (double) N + 1;
    }
}

void initialize_vector_value(double* vector, double value) {
    for (int i = 0; i < N; i++) {
        vector[i] = value;
    }
}

double norm(double* vector, int size) {
    double init_norm = 0;
    for (int i = 0; i < size; i++) {
        init_norm += vector[i] * vector[i];
    }

    return sqrt(init_norm);
}

void sub(double* a, double* b, int size, int shift) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] - b[shift + i];
    }
}

void matrix_mult(double* matrix, double* vector, int m, double* result) {
    for (int i = 0; i < m; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += matrix[i * m + j] * vector[j];
        }

        result[i] = sum;
    }
}

void const_mult(double n, double* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] *= n;
    }
}

bool is_solved(double* A, double* b, double* x, int m, int shift) {
    double* result = create_vector(m);
    matrix_mult(A, x, m, result);
    sub(result, b, m, shift);

    double result_norm = norm(result, m);
    double b_norm = norm(b, m);

    free(result);

    if ((result_norm / b_norm) < EPSILON) {
        return true;
    }

    return false;
}

void new_x(double* x, double* result, int size, int shift) {
    for (int i = 0; i < size; i++) {
        x[shift + i] -= result[i];
    }
}

double* solution(double* A, double* b, int rows_num, int wsize, int shift, int* nums, int* inds) {
    double* x = (double*) malloc(N * sizeof(double));
    initialize_vector_value(x, 0.0);

    while(!is_solved(A, b, x, rows_num, shift)) {
        double* result = create_vector(rows_num);
        matrix_mult(A, x, rows_num, result);
        sub(result, b, rows_num, shift);
        const_mult(TAU, result, rows_num);

        new_x(x, result, rows_num, shift);

        double* share_x = (double*) malloc(N * sizeof(double));
        memcpy(share_x, x + shift, rows_num * sizeof(double));
        MPI_Allgatherv(share_x, rows_num, MPI_DOUBLE, x, nums, inds, MPI_DOUBLE, MPI_COMM_WORLD);
        free(share_x);

        free(result);
    }

    return x;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // const int N = 15000;
    
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    int wsize;
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    double* b = (double*) malloc(N * sizeof(double));
    if (wrank == 0) {
        initialize_vector(b);
    }
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int* nums = (int*) malloc(wsize * sizeof(int));
    int* inds = (int*) malloc(wsize * sizeof(int));

    for (int i = 0; i < wsize; i++) {
        nums[i] = N / wsize + ((i < (N % wsize)) ? 1 : 0);
    }

    inds[0] = 0;
    for (int i = 0; i < wsize - 1; i++) {
        inds[i + 1] = nums[i] + inds[i]; 
    }

    int rows_num = nums[wrank];

    double* A = create_matrix(rows_num);
    initialize_matrix(A, rows_num, inds[wrank]);

    double min_time;

    double start, stop;

    start = MPI_Wtime();

    double* result = solution(A, b, rows_num, wsize, inds[wrank], nums, inds);

    stop = MPI_Wtime();

    double total_time = stop - start;
    MPI_Reduce(&total_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (wrank == 0) {
        printf("min time = %f\n", min_time);
    }

    free(A);
    free(b);
    free(result);
    free(nums);
    free(inds);

    MPI_Finalize();

}
