#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct Matrix {
    int size;
    double *data;
} Matrix;

Matrix MatrixCreate(int size) {
    Matrix M;
    M.size = size;
    M.data = malloc(size * size * sizeof(double));
    return M;
}

void MatrixDestroy(Matrix M) { free(M.data); }

typedef struct Vector {
    int size;
    double *data;
} Vector;

Vector VectorCreate(int size) {
    Vector V;
    V.size = size;
    V.data = malloc(size * sizeof(double));
    return V;
}

void VectorDestroy(Vector V) { free(V.data); }

Vector VectorCreateInitialized(int size, double value) {
    Vector V = VectorCreate(size);
    for (int i = 0; i < size; i++) {
        V.data[i] = value;
    }

    return V;
}

double VectorEuclideanNorm(Vector V) {
    double norm_sq = 0;
    for (int i = 0; i < V.size; i++) {
        norm_sq += V.data[i] * V.data[i];
    }

    return sqrt(norm_sq);
}

Vector MatrixVectorMult(Matrix A, Vector x) {
    int size = x.size;
    Vector res = VectorCreate(size);
    for (int i = 0; i < size; i++) {
        double sum = 0;
        for (int j = 0; j < size; j++) {
            sum += A.data[i * size + j] * x.data[j];
        }

        res.data[i] = sum;
    }

    return res;
}

void VectorSubtractInPlace(Vector a, Vector b) {
    for (int i = 0; i < a.size; i++) {
        a.data[i] -= b.data[i];
    }
}

void VectorMultiplyInPlace(Vector v, double scalar) {
    for (int i = 0; i < v.size; i++) {
        v.data[i] *= scalar;
    }
}

// Task1-specific initialization
void InitializeMatrix(Matrix M) {
    for (int i = 0; i < M.size * M.size; i++) {
        M.data[i] = 1.0;
    }

    for (int i = 0; i < M.size; i++) {
        M.data[i * M.size + i] = 2.0;
    }
}

void InitializeVector(Vector V) {
    for (int i = 0; i < V.size; i++) {
        V.data[i] = (double)V.size + 1;
    }
}

// Algorithm
Vector SolveLinearSystem(Matrix A, Vector b, double rate, double epsilon) {
    const double b_norm = VectorEuclideanNorm(b);
    Vector x = VectorCreateInitialized(b.size, 0.0);

    // int cnt = 0;

    while (1) {
        // printf("%d\n", cnt);
        // cnt++;

        Vector diff = MatrixVectorMult(A, x);
        VectorSubtractInPlace(diff, b);
        double norm = VectorEuclideanNorm(diff);

        printf("%f\n", norm);

        if (norm < epsilon * b_norm) {
            
            VectorDestroy(diff);
            break;
        }

        VectorMultiplyInPlace(diff, rate);
        VectorSubtractInPlace(x, diff);
        VectorDestroy(diff);
    }

    return x;
}

int main() {
    const int N = 1024;
    const double EPSILON = 0.00001; // 10^-5
    const double RATE = 0.01;

    Matrix A = MatrixCreate(N);
    Vector b = VectorCreate(N);
    InitializeMatrix(A);
    InitializeVector(b);

    Vector x = SolveLinearSystem(A, b, RATE, EPSILON);

    for (int i = 0; i < x.size; i++) {
        printf("%f ", x.data[i]);
    }

    MatrixDestroy(A);
    VectorDestroy(b);
    VectorDestroy(x);

    return 0;
}
