#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Define matrix dimensions

#define MATRIX_SIZE 512
//#define MATRIX_SIZE 1024
//#define MATRIX_SIZE 2048
#define NUM_COLUMNS MATRIX_SIZE // Number of columns in Matrix A
#define NUM_ROWS MATRIX_SIZE // Number of rows in Matrix A

#define NUM_THREADS 16
#define TILE_SIZE 16

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


#endif // _MATRIXMUL_H_

