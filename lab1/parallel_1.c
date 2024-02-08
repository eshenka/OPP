#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <math.h>

#define EPSILON 0.01
#define TAU 0.00001

double* create_vector(int size) {
    double* vector = (double*) malloc(size * sizeof(double));
    return vector;
}

void initialize_matrix(double* matrix, int n, int m, int wrank) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * n + j] = 1.0;
        }
        matrix[i * n + i + wrank] = 2.0;
    }
}

void initialize_vector(double* vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (double) n + 1;
    }
}

void initialize_vector_value(double* vector, double value, int size) {
    for (int i = 0; i < size; i++) {
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

double* sub(double* a, double* b, int size, int wrank) {
    double* result = create_vector(size);
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[wrank + 1 + i];
    }
    return result;
}

double* matrix_mult(double* matrix, double* vector, int m, int n) {
    double* result = create_vector(m);
    for (int i = 0; i < m; i++) {
        int sum = 0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * m + j] * vector[j];
        }

        result[i] = sum;
    }
    return result;
}

void const_mult(double n, double* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] *= n;
    }
}

bool is_solved(double* A, double* b, double* x, int m, int n, int wrank) {
    double* result = create_vector(m);
    result = matrix_mult(A, x, m, n);
    result = sub(result, b, m, wrank);

    double result_norm = norm(result, m);
    double b_norm = norm(b, n); //idk which norm is OK

    free(result);

    printf("%f\n", result_norm / b_norm);

    if ((result_norm / b_norm) < EPSILON) {
        return true;
    }

    return false;
}

void new_x(double* x, double* result, int size) {
    for (int i = 0; i < size; i++) {
        x[i] -= result[i];
    }
}

// double* result = solution(A, b, vec_size, rows_num, wsize, wrank);
double* solution(double* A, double* b, int vec_size, int rows_num, int wsize, int wrank) {
    double* x = (double*) malloc(rows_num * sizeof(double));
    initialize_vector_value(x, 0.0, rows_num);

    while(!is_solved(A, b, x, rows_num, vec_size, wrank)) {
        double* result = create_vector(rows_num);
        result = matrix_mult(A, x, rows_num, vec_size);
        result = sub(result, b, rows_num, wrank);
        const_mult(TAU, result, rows_num);

        new_x(x, result, rows_num);

        // printf("%f %d\n", x[0], wrank);

        free(result);
    }

    return x;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const int vec_size = 64;
    
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    int wsize;
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    double* b = (double*) malloc(vec_size * sizeof(double));
    if (wrank == 0) {
        initialize_vector(b, vec_size);
    }
    MPI_Bcast(b, vec_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rows_num = vec_size / wsize + ((wrank < (vec_size % wsize)) ? 1 : 0);

    // printf("rows_num = %d, wrank = %d\n", rows_num, wrank);

    double* A = (double*) malloc(vec_size * rows_num * sizeof(double));
    initialize_matrix(A, vec_size, rows_num, wrank);

    double* result = solution(A, b, vec_size, rows_num, wsize, wrank);

    double* x = create_vector(vec_size);

    int* nums = (int*) malloc(wsize * sizeof(int));
    int* inds = (int*) malloc(wsize * sizeof(int));

    for (int i = 0; i < wsize; i++) {
        nums[i] = vec_size / wsize + ((wrank < (vec_size % wsize)) ? 1 : 0);
    }

    inds[0] = 0;
    for (int i = 0; i < wsize - 1; i++) {
        inds[i + 1] = nums[i] + inds[i]; 
    }

    MPI_Allgatherv(result, rows_num, MPI_DOUBLE, x, nums, inds, MPI_DOUBLE, MPI_COMM_WORLD);

    printf("%f\n", x[0]);

    MPI_Finalize();

}
