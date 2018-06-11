#ifndef _MATRIX_H_
#define _MATRIX_H_

#define THRESHOLD 1e-5      // Threshold for convergence
#define MIN_NUMBER 2        // Min number in the A and b matrices
#define MAX_NUMBER 10       // Max number in the A and b matrices

#define THREAD_BLOCK_SIZE 128            // Size of a thread block
#define NUM_BLOCKS 32                    // Number of thread blocks

// Dimension of the n x n matrix
#define MATRIX_SIZE 1024
#define NUM_COLUMNS MATRIX_SIZE         // Number of columns in Matrix A
#define NUM_ROWS MATRIX_SIZE            // Number of rows in Matrix A

// Matrix Structure declaration
typedef struct Matrix {
	//width of the matrix represented
    unsigned int num_columns;
	//height of the matrix represented
    unsigned int num_rows;
	//number of elements between the beginnings of adjacent
	// rows in the memory layout (useful for representing sub-matrices)
    unsigned int pitch;
	//Pointer to the first element of the matrix represented
    float* elements;

    unsigned int thread_id;
} Matrix;


#endif // _MATRIX_H_

