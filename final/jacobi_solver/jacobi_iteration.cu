/* Host side code for the Jacobi method of solving a system of linear equations 
   by iteration.

   Author: Naga Kandasamy
   Date modified: 3/9/2018

   Compile as follows: make clean && make
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

// Include the kernel code
#include "jacobi_iteration_kernel.cu"

// Uncomment the line below if you want the code to spit out some debug information 
// #define DEBUG

// Prototypes of external functions called in this file
extern "C" void compute_gold (const Matrix, Matrix, const Matrix);
extern "C" void display_jacobi_solution (const Matrix, const Matrix, const Matrix);

// Prototypes of local functions used in this file
Matrix allocate_matrix_on_gpu (const Matrix);
Matrix allocate_matrix (int, int, int);
int check_if_diagonal_dominant (const Matrix);
Matrix create_diagonally_dominant_matrix (unsigned int, unsigned int);
void copy_matrix_to_device (Matrix, const Matrix);
void copy_matrix_from_device (Matrix, const Matrix);
void compute_on_device (const Matrix, Matrix, const Matrix);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
void checkCUDAError (const char *);
int checkResults( float *, float *, int, float);


int 
main(int argc, char** argv) 
{
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}

    Matrix  A;                   // The NxN constant matrix
	Matrix  B;                  // The Nx1 input matrix
	Matrix reference_x;         // The reference solution 
	Matrix gpu_solution_x;      // The solution computed by the GPU

	// Initialize the random number generator with a seed value
	srand(time(NULL));

	// Create the diagonally dominant matrix 
	A = create_diagonally_dominant_matrix (MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL){
        printf ("Error creating matrix. \n");
        exit (0);
	}
	
    B = allocate_matrix (MATRIX_SIZE, 1, 1);             // Create a matrix B holding the constants
	reference_x = allocate_matrix (MATRIX_SIZE, 1, 0);    // Create a matrix for the reference solution 
	gpu_solution_x = allocate_matrix (MATRIX_SIZE, 1, 0);  // Create a matrix for the GPU solution 

#ifdef DEBUG
	print_matrix (A);
	print_matrix (B);
	print_matrix (reference_x);
#endif

    // Compute the Jacobi solution on the CPU
	printf("Performing Jacobi iteration on the CPU. \n");
    compute_gold (A, reference_x, B);
    display_jacobi_solution(A, reference_x, B); // Display statistics
	
	// Compute the Jacobi solution on the GPU. The solution is returned in gpu_solution_x
    printf("\n Performing Jacobi iteration on the GPU. \n");
	compute_on_device (A, gpu_solution_x, B);
    display_jacobi_solution(A, gpu_solution_x, B); // Display statistics
	
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_solution_x.elements);
	
    exit(0);
}


// Complete this function to perform the Jacobi calculation on the GPU
void 
compute_on_device(const Matrix A, Matrix gpu_solution_x, const Matrix B){

}

// Allocate a device matrix of same size as M.
Matrix 
allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
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
copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void 
print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
				printf("Line number = %d ############## \n", i);
	for(unsigned int j = 0; j < M.num_columns; j++){

			printf("%f ", M.elements[i*M.num_rows + j]);
			}
		printf("\n");
	} 
	printf("\n");
	printf("####################################### \n");
}

// Returns a random floating-point number between the specified min and max values 
float 
get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

// Check for errors in kernel execution
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

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++){
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
			printf("error at %d \n",i);
			printf("element r %f and g %f \n",reference[i] ,gpu_result[i]);
            break;
        }
	}
	int maxEle;
    for(int i = 0; i < num_elements; i++){
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
			maxEle=i;
		
        }
	}
    printf("Max epsilon = %f at i = %d value at cpu %f and gpu %f \n", epsilon,maxEle,reference[maxEle],gpu_result[maxEle]); 
    return checkMark;
}


/* Function checks if the matrix is diagonally dominant. */
int
check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		if(diag_element <= sum)
			return 0;
	}

	return 1;
}

/* Create a diagonally dominant matix. */
Matrix 
create_diagonally_dominant_matrix (unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *) malloc (size * sizeof (float));

	// Create a matrix with random numbers between [-.5 and .5]
    unsigned int i, j;
	printf ("Creating a %d x %d matrix with random numbers between [-.5, .5]...", num_rows, num_columns);
	for(i = 0; i < size; i++)
		// M.elements[i] = ((float)rand ()/(float)RAND_MAX) - 0.5;
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
       	printf("done. \n");

	
	// Make the diagonal entries large with respect to the entries on each row
	printf("Generating the positive definite matrix.");
	for (i = 0; i < num_rows; i++){
		float row_sum = 0.0;		
		for(j = 0; j < num_columns; j++){
			row_sum += fabs (M.elements[i * M.num_rows + j]);
		}
		M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

	if(!check_if_diagonal_dominant (M)){
		free (M.elements);
		M.elements = NULL;
	}
	
    return M;
}



