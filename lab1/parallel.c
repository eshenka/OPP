#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define N 8
#define EPSILON 0.000001
#define TAU 0.00001

typedef struct {
    double* matrix;
} Matrix;

typedef struct {
    int index;
    int len;
    double* vector;
} Vector;

Matrix create_matrix(int M) {
    Matrix A;
    A.matrix = (double*) malloc(M * N * sizeof(double));
    
    return A;
}

void initialize_matrix(Matrix A, int M, int index) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A.matrix[i * N + j] = 1.0;
        }
    }

    for (int i = 0; i < M; i++) {
        A.matrix[i * M + i + index] = 2.0;
    }
}

Vector create_vector(int index, int len, int size) {
    Vector v;

    v.index = index; //indices[wrank]
    v.len = len; //M
    v.vector = (double*) malloc(size * sizeof(double));

    return v;
}

void initialize_vector(Vector v, double value) {
    for (int i = 0; i < v.len; i++) {
        v.vector[i] = value;
    }
}

int* create_sizes(int wsize) {
    int* sizes = (int*) malloc(wsize * sizeof(int));

    for (int i = 0; i < wsize; i++) {
        sizes[i] = (N / wsize) + (i < (N % wsize) ? 1 : 0);
    }
    
    return sizes;
}

int* create_indices(int* sizes, int wsize) {
    int* indices = (int*) malloc(wsize * sizeof(int));

    indices[0] = 0;
    for (int i = 1; i < wsize; i++) {
        indices[i] = sizes[i - 1] + indices[i - 1]; 
    }


    return indices;
}

void multiply_matrix_by_vector(Matrix A, Vector x, Vector result, int M, int wrank, int wsize, int* sizes) {
    for (int k = 0; k < wsize; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = x.index; j < x.index + x.len; j++) {
                result.vector[i] += A.matrix[i * N + j] * x.vector[j];
            }
        }
        MPI_Sendrecv_replace(x.vector, sizes[0], MPI_DOUBLE, 
                            (wrank + 1) % wsize, 123, 
                            (wrank + wsize - 1) % wsize,
                            123, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        x.index += x.len;
        x.index %= N;
        x.len = sizes[(wrank + k + 1) % wsize];
    }
}

void sub_vectors(Vector result, Vector b) {
    for (int i = 0; i < result.len; i++) {
        result.vector[i] -= b.vector[i];
    }
}

double sum_of_squares(Vector v) {
    double sum = 0;
    for (int i = 0; i < v.len; i++) {
        sum += v.vector[i] * v.vector[i];
    }

    return sum;
}

void mult_vector_by_TAU(Vector result, Vector tau_result) {
    for (int i = 0; i < result.len; i++) {
        tau_result.vector[i] = result.vector[i] * (double) TAU;
    }
}

bool is_solved(Matrix A, Vector x, Vector b, Vector result, int M, int wrank, int wsize, int* sizes) {
    multiply_matrix_by_vector(A, x, result, M, wrank, wsize, sizes); //Ax
    sub_vectors(result, b); //Ax - b
    
    double kostyl = 0;

    double sum = sum_of_squares(result);
    MPI_Allreduce(&sum, &kostyl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    double result_norm = sqrt(sum);
    double b_norm = sqrt(N * (N + 1) * (N + 1));

    if (result_norm / b_norm < EPSILON) {
        return true;
    }
    return false;

}

Vector solution(Matrix A, Vector x, Vector b, int M, int wrank, int wsize, int* sizes) {
    Vector result = create_vector(b.index, b.len, b.len);
    initialize_vector(result, 0.0);

    int n = 10;

    while (!is_solved(A, x, b, result, M, wrank, wsize, sizes) && n--) {
        Vector tau_result = create_vector(b.index, b.len, b.len);
        mult_vector_by_TAU(result, tau_result);
        sub_vectors(result, tau_result);
        free(tau_result.vector);
        memcpy(x.vector, result.vector, result.len);

        initialize_vector(result, 0.0);
    }

    return result;
}

void Organize() {
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    int* sizes = create_sizes(wsize);
    int* indices = create_indices(sizes, wsize);

    int M = sizes[wrank];

    Matrix A = create_matrix(M);
    Vector b = create_vector(indices[wrank], M, M);
    Vector x = create_vector(indices[wrank], M, sizes[0]);

    initialize_matrix(A, M, b.index);
    double value = (double) N + 1;
    initialize_vector(b, value);
    initialize_vector(x, 0.0);

    x = solution(A, x, b, M, wrank, wsize, sizes);

    MPI_Gatherv(x.vector, sizes[wrank], MPI_DOUBLE, MPI_IN_PLACE, sizes, indices, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (wrank == 0) printf("%f\n", x.vector[0]);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Organize();

    MPI_Finalize();
    return 0;
}