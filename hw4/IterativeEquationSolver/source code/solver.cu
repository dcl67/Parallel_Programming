/* 
Code for the equation solver. 
Author: Naga Kandasamy 
Date modified: 3/4/2018 
*/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h" // This file defines the grid data structure
#include <limits.h>
#include <sys/time.h>

// Include the kernel code during the preprocessing step
texture<float> tElements;
#include "solver_kernel.cu"

extern "C" void compute_gold(GRID_STRUCT *);


/* This function prints the grid on the screen */
void 
display_grid(GRID_STRUCT *my_grid)
{
	for(int i = 0; i < my_grid->dimension; i++)
		for(int j = 0; j < my_grid->dimension; j++)
			printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
		printf("\n");
}


/* This function prints out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
		// Print statistics for the CPU grid
		float min = INFINITY;
		float max = 0.0;
		double sum = 0.0; 
		for(int i = 0; i < my_grid->dimension; i++){
			for(int j = 0; j < my_grid->dimension; j++){
				sum += my_grid->element[i * my_grid->dimension + j]; // Compute the sum
				if(my_grid->element[i * my_grid->dimension + j] > max) max = my_grid->element[i * my_grid->dimension + j]; // Determine max
				if(my_grid->element[i * my_grid->dimension + j] < min) min = my_grid->element[i * my_grid->dimension + j]; // Determine min
				 
			}
		}

	printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}

/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2)
{
    double diff;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff/num_elements);
}

/* This function creates a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_for_cpu, GRID_STRUCT *grid_for_gpu)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_for_cpu->dimension, grid_for_cpu->dimension);
	grid_for_cpu->element = (float *)malloc(sizeof(float) * grid_for_cpu->num_elements);
	grid_for_gpu->element = (float *)malloc(sizeof(float) * grid_for_gpu->num_elements);


	srand((unsigned)time(NULL)); // Seed the the random number generator 
	
	float val;
	for(int i = 0; i < grid_for_cpu->dimension; i++)
		for(int j = 0; j < grid_for_cpu->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE; // Obtain a random value
			grid_for_cpu->element[i * grid_for_cpu->dimension + j] = val; 	
			grid_for_gpu->element[i * grid_for_gpu->dimension + j] = val; 				
		}
}


/* Edit this function skeleton to solve the equation on the device. Store the results back in the my_grid->element data structure for comparison with the CPU result. */
void 
compute_on_device(GRID_STRUCT *my_grid)
{
	int j, done = 0;
	float diff = 0.0f;
	float time;

	float* grid1_h = (float *)malloc(sizeof(float) * my_grid->num_elements);
	float* grid1 = NULL;
	cudaMalloc((void**) &grid1, sizeof(float) * my_grid->num_elements);
	cudaMemcpy(grid1, my_grid->element, sizeof(float) * my_grid->num_elements, cudaMemcpyHostToDevice);
	
	float* grid21 = (float *)malloc(sizeof(float) * my_grid->num_elements);
	float* grid2 = NULL;
	cudaMalloc((void**) &grid2, sizeof(float) * my_grid->num_elements);
	cudaMemset(grid2, 0.0f, my_grid->num_elements);

	float* diff1 = (float *)malloc(sizeof(float) * my_grid->num_elements);
	float* diff2 = NULL;
	cudaMalloc((void**) &diff2, sizeof(float) * my_grid->num_elements);
	cudaMemset(diff2, 0.0f, my_grid->num_elements);
	
	dim3 dimBlock(TILE_SIZE,TILE_SIZE);
	dim3 dimGrid(GRID_DIMENSION/TILE_SIZE, GRID_DIMENSION/TILE_SIZE);

	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
	while(!done)
	{
		solver_kernel_naive<<< dimGrid, dimBlock >>>(grid1,grid2, GRID_DIMENSION, diff2);
		cudaThreadSynchronize();
		float* temp = grid1;
		grid1 = grid2;
		grid2 = temp;
		cudaMemcpy(diff1, diff2, sizeof(float)*my_grid->num_elements, cudaMemcpyDeviceToHost);
		for(int i = 0; i < my_grid->num_elements; i++){
			diff += diff1[i];
		}
		if(diff/(my_grid->num_elements) < (float)TOLERANCE){
			done = 1;
		}
		diff=0;
		j++;
	}
	
	printf("Convergence achieved on the GPU using global memory after %d iterations. \n", j);
	gettimeofday(&stop, NULL);
	time = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("GPU Global run time = %f s. \n\n", time);
	
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){ 
		fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
		return;
	}

	cudaFree(grid1);
	cudaFree(grid2);
	cudaFree(diff2);
	cudaUnbindTexture(inputTex1D);
	cudaUnbindTexture(outputTex1D);
	cudaUnbindTexture(diffTex1D); 

	free(grid21);
}


void
compute_on_device_optimized(GRID_STRUCT *my_grid){
	float *deviceElements = NULL;
	float *deviceDifferences = NULL;
	int *mutex = NULL;
	float tmp = INT_MAX;
	unsigned int j = 0;

	cudaMalloc((void**)&deviceElements, my_grid->num_elements * sizeof(float));
	cudaMalloc((void**)&deviceDifferences, sizeof(float));
    cudaMalloc((void **)&mutex, sizeof(int));

	cudaMemset(deviceDifferences, 0, sizeof(float));
    cudaMemset(mutex, 0, sizeof(int));
	cudaMemcpy(deviceElements, my_grid->element, my_grid->num_elements * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaBindTexture(NULL, tElements, deviceElements, my_grid->num_elements * sizeof(float));

	dim3 thread_block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(1,(my_grid->dimension) / TILE_SIZE);

    struct timeval start, stop;
    gettimeofday(&start, NULL);
	
	while (tmp / (my_grid->dimension * my_grid->dimension) >= TOLERANCE){
		cudaMemset(deviceDifferences, 0, sizeof(float));
		solver_kernel_optimized<<<grid, thread_block>>>(deviceElements, deviceDifferences, mutex);
		cudaThreadSynchronize();

		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err){ 
			fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
			return;
		}

		cudaMemcpy(&tmp, deviceDifferences, sizeof(float), cudaMemcpyDeviceToHost);
		j++;
	}

	gettimeofday(&stop, NULL);
    printf("Iterations: %d, GPU Shared Memory runtime: %f s.\n\n", j, (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	cudaMemcpy(my_grid->element, deviceElements, my_grid->num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(deviceElements);
	cudaFree(deviceDifferences);
}


/* The main function */
int 
main(int argc, char **argv)
{	
	/* Generate the grid */
	GRID_STRUCT *grid_for_cpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure

	grid_for_cpu->dimension = GRID_DIMENSION;
	grid_for_cpu->num_elements = grid_for_cpu->dimension * grid_for_cpu->dimension;
	grid_for_gpu->dimension = GRID_DIMENSION;
	grid_for_gpu->num_elements = grid_for_gpu->dimension * grid_for_gpu->dimension;

 	create_grids(grid_for_cpu, grid_for_gpu); // Create the grids and populate them with the same set of random values
	
	printf("Using the cpu to solve the grid. \n");
	struct timeval start, stop;
	gettimeofday(&start, NULL);
	compute_gold(grid_for_cpu);  // Use CPU to solve 
	gettimeofday(&stop, NULL);
	printf("\nCPU serial run time= %f s. \n\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	// Use the GPU to solve the equation
	struct timeval start1, stop1;
	gettimeofday(&start1, NULL);
	compute_on_device(grid_for_gpu);
	gettimeofday(&stop1, NULL);
	//printf("\nGPU global run time= %f s. \n", (float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000));
	
	// Use the GPU with shared memory
	struct timeval start2, stop2;
	gettimeofday(&start2, NULL);
	compute_on_device_optimized(grid_for_gpu);
	gettimeofday(&stop2, NULL);
	//printf("\nGPU shared memory run time= %f s. \n", (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000));

	// Print key statistics for the converged values
	printf("CPU: \n");
	print_statistics(grid_for_cpu);

	printf("GPU: \n");
	print_statistics(grid_for_gpu);
	
    /* Compute grid differences. */
    compute_grid_differences(grid_for_cpu, grid_for_gpu);

	free((void *)grid_for_cpu->element);	
	free((void *)grid_for_cpu); // Free the grid data structure 
	
	free((void *)grid_for_gpu->element);	
	free((void *)grid_for_gpu); // Free the grid data structure 

	exit(0);
}
