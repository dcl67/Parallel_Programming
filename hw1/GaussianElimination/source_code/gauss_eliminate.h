#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#define MATRIX_SIZE 1024
//#define MATRIX_SIZE 2048
//#define MATRIX_SIZE 4096
//#define MATRIX_SIZE 8192
#define NUM_THREADS 8

#define NUM_COLUMNS MATRIX_SIZE             /* Number of columns in Matrix A. */
#define NUM_ROWS MATRIX_SIZE                /* Number of rows in Matrix A. */

/* Matrix Structure declaration. */
typedef struct {
    unsigned int num_columns;               /* Width of the matrix represented. */
    unsigned int num_rows;                  /* Height of the matrix represented. */
	
    /* Mumber of elements between the beginnings of adjacent rows in the memory layout.
     * This is useful for representing sub-matrices. 
     * */
    unsigned int pitch;
    float* elements;                        /* Pointer to the first element of the matrix represented. */

} Matrix;

#endif // _MATRIXMUL_H_

