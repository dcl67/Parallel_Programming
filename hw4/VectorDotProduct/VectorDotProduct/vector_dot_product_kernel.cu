#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_
#define BLOCK_SIZE 1024
#define GRID_SIZE 1024

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */
__device__ void lock(int *mutex);
__device__ void unlock(int *mutex);

__global__ void vector_dot_product_kernel(int num_elements, float* a, float* b, float* result, int *mutex)
{
	__shared__ float sum[BLOCK_SIZE];
	float thread_sum = 0.0;
	int tx = threadIdx.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i = tid;
	int slice = blockDim.x * gridDim.x;

	while(i < num_elements){
		thread_sum += a[i] * b[i];
		i += slice;
	}

	sum[threadIdx.x] = thread_sum;
	__syncthreads();

	for(int slice = blockDim.x/2; slice > 0; slice /= 2){
		if(tx < slice)
			sum[tx] += sum[tx+slice];
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		lock(mutex);
		result[0] += sum[0];
		unlock(mutex);
	}
}

__device__ void lock(int *mutex){
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex){
	atomicExch(mutex, 0);
}

#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
