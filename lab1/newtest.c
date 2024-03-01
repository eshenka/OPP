#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    double* x = (double*) malloc(4 * sizeof(double));

    int end = wrank == 0 ? 3 : 4;

    for (int i = 0; i < end; i++) {
        x[i] = end;
    }

    MPI_Sendrecv_replace(x, 4, MPI_DOUBLE, 
                        (wrank + 1) % wsize, 123, 
                        (wrank + wsize - 1) % wsize,
                        123, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    MPI_Sendrecv_replace(x, 4, MPI_DOUBLE, 
                        (wrank + 1) % wsize, 123, 
                        (wrank + wsize - 1) % wsize,
                        123, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    printf("%f %d\n", x[0], wrank);

    MPI_Finalize();
    return 0;
}