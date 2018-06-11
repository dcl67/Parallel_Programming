#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// includes, kernels
#include "vector_dot_product_kernel.cu"

void run_test(unsigned int);
float compute_on_device(float *, float *,int);
void check_for_error(char *);
extern "C" float compute_gold( float *, float *, unsigned int);

int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	
	printf("Generating dot product on the CPU. \n");
	struct timeval start1, stop1;

    gettimeofday(&start1, NULL);
	float reference = compute_gold(A, B, num_elements);
	gettimeofday(&stop1, NULL);
	printf("CPU execution time = %fs. \n", (float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000));
    
	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	float gpu_result = compute_on_device(A, B, num_elements);

	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    printf("Epsilon: %f. \n", fabsf(reference - gpu_result));

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

/* Edit this function to compute the dot product on the device using atomic intrinsics. */
float 
compute_on_device(float *A_on_host, float *B_on_host, int num_elements)
{
    //return 0;
    float result = 0.0;
	float *A2 = NULL;
	float *B2 = NULL;
	float *C2 = NULL;

	cudaMalloc((void**)&A2, num_elements*sizeof(float));
	cudaMemcpy(A2, A_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B2, num_elements*sizeof(float));
	cudaMemcpy(B2, B_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&C2, GRID_SIZE*sizeof(float));
	cudaMemset(C2, 0.0, GRID_SIZE*sizeof(float));
	
	int *mutex = NULL;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(GRID_SIZE, 1);

	struct timeval start2, stop2;

	gettimeofday(&start2, NULL);

	vector_dot_product_kernel <<< dimGrid, dimBlock >>> (num_elements, A2, B2, C2, mutex);
	cudaThreadSynchronize();

	gettimeofday(&stop2, NULL);
	printf("GPU execution time = %fs. \n", (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000));

	check_for_error("Error in kernel");
	cudaMemcpy(&result, C2, sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(A2);
	cudaFree(B2);
	cudaFree(C2);
	
	return result;
}
 
// This function checks for errors returned by the CUDA run time
void 
check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
