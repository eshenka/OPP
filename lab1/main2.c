#include <stdio.h>

typedef struct {
    int begin;
    int end;
} Chunk;

Chunk get_things(int size, int wsize, int wrank) {
    Chunk chunk;
    chunk.begin = wrank * (size / wsize) + (wrank < size % wsize ? wrank : size % wsize);
    chunk.end = chunk.begin + size / wsize + (wrank < size % wsize ? 1 : 0);
    return chunk;
}

int main() {
    int size = 8192;
    int wsize = 6;
    for (int i = 0; i < wsize; i++) {
        Chunk chunk = get_things(size, wsize, i);
        printf("%d %d elements = %d\n", chunk.begin, chunk.end, chunk.end - chunk.begin);
    }
}

// int main() {
//     int size = 14;
//     int rank = 0;
//     int world_size = 2;

//     int matrix_size;

//     if (rank < size % world_size) {
//         matrix_size = size / world_size;
//     } else {
//         matrix_size = size / world_size - 1;
//     }

//     printf("matrix size %d\n", matrix_size);

//     if (rank <= size % world_size) {

//         int beg = rank * (size / world_size + 1);
//         int end = beg + matrix_size;

//         printf("%d %d", beg, end);
//     } else {
//         int beg = (size % world_size) * (matrix_size + 1) + (rank - size % world_size) * matrix_size + rank;
//         int end = beg + matrix_size;

//         printf("%d %d", beg, end);
//     }
// }
