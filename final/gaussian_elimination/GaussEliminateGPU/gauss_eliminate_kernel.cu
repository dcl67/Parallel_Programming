 /* Device code. */
#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float *U, int k)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ double shared1[THREAD_BLOCK_SIZE];
	__shared__ double U_shared;

	// Store elements U[k,k] into shared memory	
	if (threadIdx.x == 0) {
		U_shared = (double)U[MATRIX_SIZE*k + k];
	}

	__syncthreads();

	if (j > k && j < MATRIX_SIZE){

		__syncthreads();

		// Division Step
		U[MATRIX_SIZE * k + j] = (double)(U[MATRIX_SIZE * k + j]/U_shared);

		shared1[threadIdx.x] = U[MATRIX_SIZE * k + j];
		
		__syncthreads();

		double tmp = shared1[j % blockDim.x];

		// Elimination step
		for (int i = k+1; i < MATRIX_SIZE; i++){
			double temp1 = U[MATRIX_SIZE * i + j];
			double temp2 = U[MATRIX_SIZE * i + k];
			__syncthreads();
			temp1 -= __fmul_rn(temp2, tmp);
			__syncthreads();
			
			U[MATRIX_SIZE * i + j] = temp1;
			__syncthreads();
		}	
	}
	
	if (j == MATRIX_SIZE-1){
		U[MATRIX_SIZE * k + k] = 1;
		for (int s = k+1; s < MATRIX_SIZE; s++){
			U[MATRIX_SIZE * s + k] = 0;	
			__syncthreads();	
		}
	}
	__syncthreads();
}

