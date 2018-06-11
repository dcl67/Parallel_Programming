#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

__device__ void lock(int *mutex){
    while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex){
    atomicExch(mutex, 0);
}

texture<float>inputTex1D;
texture<float>diffTex1D;
texture<float>outputTex1D;

__global__ void solver_kernel_naive(float* input, float* output, int N, float* globalDiff){
	
	unsigned int tx = threadIdx.x;
	unsigned int x = blockIdx.x * blockDim.x + tx;
	
	unsigned int ty = threadIdx.y;
	unsigned int y = blockIdx.y * blockDim.y + ty;
		
	if(x > 0 && y > 0 && x < (N-1) && y < (N-1)) {
		output[x*N + y] = 0.20f * (input[x*N + y] + input[(x-1)*N +y] + input[(x+1)*N +y] +\
			input[x*N + (y-1)] + input[x*N + (y+1)]);
	}
	else
		output[x*N+y] = input[x*N+y];
			
	globalDiff[x*N+y] = fabsf(output[x*N + y] - input[x*N + y]); 
}

__global__ void solver_kernel_optimized(float * element, float * global_diff, int * mutex)
{
    __shared__ float tRes[TILE_SIZE * TILE_SIZE];
    
    int index = 0.0f;
    float tmp = 0.0f;
    float diff = 0.0f;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = threadIdx.x;
    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;

    for (int i = column; i < GRID_DIMENSION; i += TILE_SIZE){
        if (i > 0 && i < GRID_DIMENSION - 1 && row > 0 && row < GRID_DIMENSION - 1){
            index = row * GRID_DIMENSION + i;
            tmp = element[index];
            element[index] = 0.20 * (tex1Dfetch(tElements, index) + tex1Dfetch(tElements, index - GRID_DIMENSION) + tex1Dfetch(tElements, index + GRID_DIMENSION) + tex1Dfetch(tElements, index + 1) + tex1Dfetch(tElements, index - 1));
            diff += fabsf(element[index] - tmp);
        }
    }
    tRes[tid] = diff;
    __syncthreads();
     
    unsigned int i = TILE_SIZE * TILE_SIZE / 2;
    while (i != 0){
    	if (tid < i){
    		tRes[tid] += tRes[tid + i];
        }
    	__syncthreads();
    	i /= 2;
    }

    if (tid == 0){
        lock(mutex);
        *global_diff += tRes[0];
        unlock(mutex);
    }
}

#endif /* _MATRIXMUL_KERNEL_H_ */
