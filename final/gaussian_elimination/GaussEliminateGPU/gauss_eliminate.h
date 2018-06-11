#ifndef _GAUSS_ELIMINATE_H_
#define _GAUSS_ELIMINATE_H_

#define THREAD_BLOCK_SIZE 64

// Matrix dimensions

#define MATRIX_SIZE 512
//#define MATRIX_SIZE 1024
//#define MATRIX_SIZE 2048

#define NUM_COLUMNS MATRIX_SIZE // Number of columns in Matrix A
#define NUM_ROWS MATRIX_SIZE // Number of rows in Matrix A

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



