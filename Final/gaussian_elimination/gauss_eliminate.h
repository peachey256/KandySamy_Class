#ifndef _GAUSS_ELIMINATE_H_
#define _GAUSS_ELIMINATE_H_

// Matrix dimensions
#define MATRIX_SIZE 2048
#define NUM_COLUMNS MATRIX_SIZE // Number of columns in Matrix A
#define NUM_ROWS MATRIX_SIZE // Number of rows in Matrix A
#define GRID_SIZE 10
#define BLOCK_SIZE 256

#define GRID_MAX  4
#define BLOCK_MAX 32


// Matrix Structure declaration
typedef struct {
	//width of the matrix represented
    unsigned int num_columns;
	//height of the matrix represented
    unsigned int num_rows;
	//number of elements between the beginnings of adjacent
	// rows in the memory layout (useful for representing sub-matrices)
    unsigned int pitch;
	//Pointer to the first element of the matrix represented
    float* elements;
} Matrix;

#endif
