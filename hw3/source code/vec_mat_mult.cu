/* Vector-matrix multiplication: Y = A * X.
 * Host code.
 * Author: Naga Kandasamy
 * Date: 2/21/2017
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "vec_mat_mult_kernel.cu"

#define MIN_NUMBER 1
#define MAX_NUMBER 4


extern "C" void compute_gold(float*, const float*, const float*, unsigned int, unsigned int);
Matrix allocate_matrix_on_gpu(const Matrix);
Matrix allocate_matrix(int, int, int);
void copy_matrix_to_device(Matrix, const Matrix);
void copy_matrix_from_device(Matrix, const Matrix);
void vec_mat_mult_on_device_using_global_memory(const Matrix, const Matrix, Matrix);
void vec_mat_mult_on_device_using_shared_memory(const Matrix, const Matrix, Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int checkResults(float *, float *, int, float);
void checkCUDAError(const char *msg);
void FreeDeviceMatrix(Matrix *);


int 
main(int argc, char** argv) {
	// Matrices for the program
	Matrix  A; // N x N matrix
	Matrix  X; // N x 1 vector
	Matrix  Y_cpu, Y_gpu_1, Y_gpu_2; // N x 1 vector
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1); // Create a random N x N matrix
	X  = allocate_matrix(MATRIX_SIZE, 1, 1); // Create a random N x 1 vector 
	Y_cpu  = allocate_matrix(MATRIX_SIZE, 1, 0); // Allocate memory for the output vectors
	Y_gpu_1 = allocate_matrix(MATRIX_SIZE, 1, 0); 
    Y_gpu_2 = allocate_matrix(MATRIX_SIZE, 1, 0);
 
    // compute the vector-matrix multiplication on the CPU for comparison    		
	//struct timeval start0, stop0;
    //gettimeofday(&start0, NULL);
    struct timeval start0, stop0;	
	gettimeofday(&start0, NULL);

	compute_gold(Y_cpu.elements, A.elements, X.elements, A.num_rows, A.num_columns);
   	gettimeofday(&stop0, NULL);
	printf("\nSERIAL:\nCPU run time= %f s. \n", (float)(stop0.tv_sec - start0.tv_sec + (stop0.tv_usec - start0.tv_usec)/(float)1000000));
	
	// Perform the vector-matrix multiplication on the GPU using global memory
    // Return the results in Y_gpu_1	
	struct timeval start1, stop1;
    gettimeofday(&start1, NULL);

	vec_mat_mult_on_device_using_global_memory(A, X, Y_gpu_1);
	gettimeofday(&stop1, NULL);
	printf("\nGLOBAL MEMORY:\nGPU run time from global memory = %f s. \n", (float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000));

	// check if the device result is equivalent to the expected solution
    printf("Checking against reference result. \n");
	int size_elements = NUM_ROWS;
	int res = checkResults(Y_cpu.elements, Y_gpu_1.elements, size_elements, 0.0001);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");


    // Perform the vector-matrix multiplication on the GPU using shared memory
    // Return the results in Y_gpu_2
    struct timeval start2, stop2;	
	gettimeofday(&start2, NULL);

	vec_mat_mult_on_device_using_shared_memory(A, X, Y_gpu_2);

    gettimeofday(&stop2, NULL);
	printf("\nSHARED MEMORY:\nGPU run time from shared memory = %f s. \n", (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000));

	// check if the device result is equivalent to the expected solution
    printf("Checking against reference result. \n");
    res = checkResults(Y_cpu.elements, Y_gpu_2.elements, size_elements, 0.0001);
	printf("Test %s\n", (1 == res) ? "PASSED\n" : "FAILED\n");

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(X.elements); X.elements = NULL;
	free(Y_cpu.elements); Y_cpu.elements = NULL;
	free(Y_gpu_1.elements); Y_gpu_1.elements = NULL;
    free(Y_gpu_2.elements); Y_gpu_2.elements = NULL;

	return 0;
}

// Complete the functionality of vector-matrix multiplication using the GPU 
// Kernel should use global memory
void 
vec_mat_mult_on_device_using_global_memory(const Matrix A, const Matrix X, Matrix Y)
{
	Matrix A2=allocate_matrix_on_gpu(A);
	Matrix X2=allocate_matrix_on_gpu(X);
	Matrix Y2=allocate_matrix_on_gpu(Y);

	copy_matrix_to_device(A2, A);
	copy_matrix_to_device(X2, X);

	dim3 threads(THREADS,1);
	dim3 grid(MATRIX_SIZE/threads.x,1);

	vec_mat_kernel_naive<<< grid, threads >>>(A2.elements,X2.elements,Y2.elements);
	cudaThreadSynchronize();

	//checkCUDAError("Error in kernel, check vec_mat_kernal_naive.\n");
	copy_matrix_from_device(Y,Y2);

	//FreeDeviceMatrix(&A2);
	cudaFree(A2.elements);
	A2.elements = NULL;
	//FreeDeviceMatrix(&X2);
	cudaFree(X2.elements);
	X2.elements = NULL;
	//FreeDeviceMatrix(&Y2);
	cudaFree(Y2.elements);
	Y2.elements = NULL;
}

// Complete the functionality of vector-matrix multiplication using the GPU
// Kernel should use shared memory
void 
vec_mat_mult_on_device_using_shared_memory(const Matrix A, const Matrix X, Matrix Y)
{
	Matrix Ad = allocate_matrix_on_gpu(A);
    copy_matrix_to_device(Ad, A);
    Matrix Xd = allocate_matrix_on_gpu(X);
    copy_matrix_to_device(Xd, X);
    Matrix Yd = allocate_matrix_on_gpu(Y);
    copy_matrix_to_device(Yd, Y);

    dim3 dimBlock, dimGrid;
    dimBlock.x = dimBlock.y = TILE_SIZE;
    dimBlock.z = 1;
    dimGrid.x = (Y.num_columns/ dimBlock.x) + ((Y.num_columns % dimBlock.x) ? 1:0 );
    dimGrid.y = (Y.num_rows / dimBlock.y) + ((Y.num_rows % dimBlock.y) ? 1:0 );
    dimGrid.z = 1;

    //struct timeval start, stop;	
	//gettimeofday(&start, NULL);

	vec_mat_kernel_optimized<<<dimGrid,dimBlock>>>(Ad,Xd,Yd); 
	cudaThreadSynchronize();

    //gettimeofday(&stop, NULL);
	//printf("\nSHARED MEMORY:\nGPU run time from shared memory = %f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    checkCUDAError("Error in kernel");  
    copy_matrix_from_device(Y, Yd); 

    FreeDeviceMatrix(&Ad);
    FreeDeviceMatrix(&Xd);
    FreeDeviceMatrix(&Yd);
}


// Allocate a device matrix of same size as M.
Matrix 
allocate_matrix_on_gpu(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix 
allocate_matrix(int num_rows, int num_columns, int init)
{
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void 
copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void 
copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void 
print_matrix(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_columns + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float 
get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
            break;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}

void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}

void 
FreeDeviceMatrix(Matrix* M)
{
	cudaFree(M->elements);
	M->elements = NULL;
}
