#include <math.h>
#include <mpi.h>
#include <mpich/mpi.h>
#include <mpich/mpi_proto.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 100
#define EPSILON 0.00000000001
#define TAU 0.00001

double* create_matrix(int m) {
    double* matrix = (double*)malloc(m * N * sizeof(double));
    return matrix;
}

double* create_vector(int size) {
    double* vector = (double*)malloc(size * sizeof(double));
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
        vector[i] = (double)N + 1;
    }
}

void initialize_vector_value(double* vector, double value) {
    for (int i = 0; i < N; i++) {
        vector[i] = value;
    }
}

double norm_sq(double* vector, int size) {
    double init_norm = 0;
    for (int i = 0; i < size; i++) {
        init_norm += vector[i] * vector[i];
    }

    return init_norm;
}

void sub(double* a, double* b, int size, int shift) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] - b[shift + i];
    }
}

void matrix_mult(double* matrix, double* vector, int m, double* result) {
    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += matrix[i * N + j] * vector[j];
        }

        result[i] = sum;
    }
}

void const_mult(double n, double* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] *= n;
    }
}

bool is_solved(double* A, double* b, double* x, double* result, int m, int shift) {
    matrix_mult(A, x, m, result);
    sub(result, b, m, shift);

    double partial_result_norm = norm_sq(result, m);
    double partial_b_norm = norm_sq(b, m);

    double result_norm = 0;
    double b_norm = 0;

    MPI_Allreduce(&partial_result_norm, &result_norm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&partial_b_norm, &b_norm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    result_norm = sqrt(result_norm);
    b_norm = sqrt(b_norm);

    // printf("norm: %lf\n", result_norm);

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

double* solution(double* A, double* b, int rows_num, int wsize, int shift,
                 int* nums, int* inds) {
    double* x = (double*)malloc(N * sizeof(double));
    initialize_vector_value(x, 0.0);
    double* result = create_vector(rows_num);

    // int iters = 0;
    while (!is_solved(A, b, x, result, rows_num, shift)) {
        // iters++;        
        const_mult(TAU, result, rows_num);

        new_x(x, result, rows_num, shift);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x, nums, inds,
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // printf("iters: %d\n", iters);
    free(result);
    return x;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // const int N = 15000;

    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    int wsize;
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    double* b = (double*)malloc(N * sizeof(double));
    if (wrank == 0) {
        initialize_vector(b);
    }
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int* nums = (int*)malloc(wsize * sizeof(int));
    int* inds = (int*)malloc(wsize * sizeof(int));

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
    MPI_Reduce(&total_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);

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

