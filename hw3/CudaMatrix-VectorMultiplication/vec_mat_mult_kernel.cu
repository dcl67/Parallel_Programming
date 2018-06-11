/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	double tmp=0.0;
	int i;
	int product=blockDim.x*blockIdx.x+threadIdx.x;
	
	for(i=0; i<MATRIX_SIZE; i++){
		double elementA=Ad[MATRIX_SIZE*product+i]; // Scan through row elements
		double elementX=Xd[i];
		tmp+=elementA*elementX; 
	}
	Yd[product]=(float)tmp;
}

/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(Matrix A, Matrix X, Matrix Y)
{
	__shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedX[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = (blockDim.y * blockIdx.y + ty);
    int column = blockDim.x * blockIdx.x + tx;
    int i = 0;
    int temp;
    double YElement = 0.0f;
  
    while(i < A.num_columns){

        if(i + tx < A.num_columns && row < A.num_rows){
            sharedA[ty][tx] = A.elements[row*A.num_columns + i + tx];
		}
        else{
            sharedA[ty][tx] = 0.0f;
		}
        if(i + threadIdx.y < X.num_rows && column < X.num_columns){
            sharedX[ty][tx] = X.elements[(i+ty)*X.num_columns + column];
        }
        else{
            sharedX[ty][tx] = 0.0f;
		}
        __syncthreads();

        for(temp = 0; temp < TILE_SIZE; temp++){
            YElement += sharedA[ty][temp] * sharedX[temp][tx];
		}
        __syncthreads();
        i += TILE_SIZE;
    }

    if(column < Y.num_columns && row < Y.num_rows){
       Y.elements[row*Y.num_columns + column] = (float)YElement;
	}
    return;
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
