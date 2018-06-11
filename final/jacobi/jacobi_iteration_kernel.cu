#ifndef _JACOBI_KERNEL_
#define _JACOBI_KERNEL_
#include "jacobi_iteration.h"
#include <stdio.h>

// Write the GPU kernel to solve the Jacobi iterations

__device__ void lock(int *mutex){
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex){
	atomicExch(mutex, 0);
}

__global__ void jacobi_iteration_kernel(Matrix A, Matrix B, double *ssd, int* mutex, Matrix X){
	double sum = 0;
	unsigned int num_rows = A.num_rows;
	unsigned int num_cols = A.num_columns;

 	__shared__ double ssd_local[THREAD_BLOCK_SIZE];

	double new_x[MATRIX_SIZE];

	int col= THREAD_BLOCK_SIZE * blockIdx.x + threadIdx.x;
	int t_id = threadIdx.y * THREAD_BLOCK_SIZE + threadIdx.x;
	
	unsigned int i;
	if (col < num_rows){
		int id_A = col * A.num_columns;
		for (int j = 0; j < num_cols; ++j)
			if (col != j)
				sum += (double)A.elements[id_A + j] * (double)X.elements[j];

		new_x[col] = (double)((double)B.elements[col] - sum) / (double)((double)A.elements[id_A + col]);
	
		__syncthreads();
		ssd_local[t_id] += (new_x[col] - (double)X.elements[col])*(new_x[col] - (double)X.elements[col]);
	
		X.elements[col] = new_x[col];
	}

	__syncthreads();

	i = THREAD_BLOCK_SIZE / 2;
	while (i != 0){
		if (t_id < i)
			ssd_local[t_id] += ssd_local[t_id + i];
		__syncthreads();
		i /= 2;
	}
	__syncthreads();
	if(t_id == 0){
		lock(mutex);
		*ssd = ssd_local[0];
		unlock(mutex);
	}
	return;
}
#endif
