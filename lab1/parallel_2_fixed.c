#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 100
#define EPSILON 0.00000000001
#define TAU 0.00001

typedef struct {
    double* data;
    int m;
    int n;
} Matrix;

typedef struct {
    double* data;
    int first;
    int size;
} Vector;

Matrix create_matrix(int m) {
    Matrix A;

    A.data = (double*)malloc(m * N * sizeof(double));
    A.m = m;
    A.n = N;

    return A;
}

void initialize_matrix(Matrix matrix, int shift) {
    for (int i = 0; i < matrix.m * matrix.n; i++) {
        matrix.data[i] = 1.0;
    }

    for (int i = 0; i < matrix.m; i++) {
        matrix.data[i * matrix.n + i + shift] = 2.0;
    }
}

Vector create_vector(int size, int actual_size, int first) {
    Vector v;

    v.data = (double*)malloc(size * sizeof(double));
    v.first = first;
    v.size = actual_size;

    return v;
}

void initialize_vector(Vector v, double value) {
    for (int i = 0; i < v.size; i++) {
        v.data[i] = value;
    }
}

void mult_matrix_by_vector_and_sub_vector(Matrix A, Vector x, Vector b,
                                          Vector result, int wrank, int wsize,
                                          int* nums, int* inds) {
    for (int k = 0; k < wsize; k++) {
        for (int i = 0; i < A.m; i++) {
            for (int j = 0; j < x.size; j++) {
                result.data[i] += A.data[i * A.n + j + x.first] * x.data[j];
            }
        }

        MPI_Sendrecv_replace(x.data, nums[0], MPI_DOUBLE, (wrank + 1) % wsize,
                             123, (wrank + wsize - 1) % wsize, 123,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        x.first = inds[(wrank + 2 * wsize - k - 1) % wsize];
        x.size = nums[(wrank + 2 * wsize - k - 1) % wsize];
    }

    for (int i = 0; i < A.m; i++) {
        result.data[i] -= b.data[i];
    }
}

void sub_vector(Vector result, Vector b) {
    for (int i = 0; i < result.size; i++) {
        result.data[i] = result.data[i] - b.data[i];
    }
}

void mult_vector_by_const(Vector result) {
    for (int i = 0; i < result.size; i++) {
        result.data[i] *= TAU;
    }
}

double norm(Vector v) {
    double sum = 0;
    for (int j = 0; j < v.size; j++) {
        sum += v.data[j] * v.data[j];
    }

    double all_sum = 0;

    MPI_Allreduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sqrt(all_sum);
}

bool is_solved(Matrix A, Vector b, Vector x, Vector result, int wrank,
               int wsize, int* nums, int* inds) {
    initialize_vector(result, 0.0);

    mult_matrix_by_vector_and_sub_vector(A, x, b, result, wrank, wsize, nums,
                                         inds);

    double numerator = norm(result);
    double denomenator = sqrt(N) * sqrt((N + 1) * (N + 1));

    // printf("norm: %lf\n", numerator);

    if (numerator / denomenator < EPSILON) {
        return true;
    }

    return false;
}

Vector solution(Matrix A, Vector b, int max_size, int wrank, int wsize,
                int* nums, int* inds) {
    Vector x = create_vector(max_size, b.size, b.first);
    initialize_vector(x, 0.0);

    Vector result = create_vector(A.m, A.m, 0);

    //int iters = 0;
    while (!is_solved(A, b, x, result, wrank, wsize, nums, inds)) {
        //iters++;

        mult_vector_by_const(result);  // TAU(A * x - b)

        sub_vector(x, result);  // x = x - TAU(A * x - b)
        // if (wrank == 0) {
        //     printf("%f\n", x.data[0]);
        // }
    }

    //printf("iters: %d\n", iters);

    free(result.data);

    return x;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    int wsize;
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    int* nums = (int*)malloc(wsize * sizeof(int));
    int* inds = (int*)malloc(wsize * sizeof(int));

    for (int i = 0; i < wsize; i++) {
        nums[i] = N / wsize + ((i < (N % wsize)) ? 1 : 0);
    }

    inds[0] = 0;
    for (int i = 0; i < wsize - 1; i++) {
        inds[i + 1] = nums[i] + inds[i];
    }

    int m = nums[wrank];
    int size = nums[0];

    Matrix A = create_matrix(m);
    initialize_matrix(A, inds[wrank]);

    Vector b = create_vector(size, m, inds[wrank]);
    initialize_vector(b, (double)N + 1);

    double min_time;

    double start, stop;

    start = MPI_Wtime();

    Vector x = solution(A, b, size, wrank, wsize, nums, inds);

    double* result = (double*)malloc(N * sizeof(double));
    MPI_Gatherv(x.data, nums[wrank], MPI_DOUBLE, result, nums, inds, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    stop = MPI_Wtime();

    double total_time = stop - start;
    MPI_Reduce(&total_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);

    if (wrank == 0) {
        printf("min time = %f\n", min_time);
    }
    
    free(A.data);
    free(b.data);
    free(nums);
    free(inds);
    free(x.data);

    MPI_Finalize();
    return 0;
}

