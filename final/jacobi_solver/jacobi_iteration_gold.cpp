/* Reference code for solving the equation by jacobi by iteration method. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi_iteration.h"

extern Matrix allocate_matrix(int num_rows, int num_columns, int init);
extern "C" void display_jacobi_solution(const Matrix A, const Matrix reference, const Matrix B);
extern "C" void compute_gold (const Matrix A, const Matrix reference, const Matrix B);

void
compute_gold (const Matrix A, Matrix ref_x, const Matrix B)
{
    unsigned int i, j, k;
    unsigned int num_rows = A.num_rows; // Rows in matrix A
    unsigned int num_cols = A.num_columns; // Columns in matrix A
    Matrix new_x = allocate_matrix (MATRIX_SIZE, 1, 0);  // Allocate n x 1 matrix to hold iteration values
    
    // Initialize the current jacobi solution
    for (i = 0; i < num_rows; i++)
        ref_x.elements[i] = B.elements[i];

    // Perform the Jacobi iteration 
    unsigned int done = 0;
    double ssd, mse;
    unsigned int num_iter = 0;
    
    while (!done){
        for (i = 0; i < num_rows; i++){
            double sum = 0.0;
            for (j = 0; j < num_cols; j++){
                if (i != j)
                    sum += A.elements[i * num_cols + j] * ref_x.elements[j];
            }
           
            // Calculate the new values for the unkowns for the current row 
            new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
        }

        /* Note: you can optimize the above nested loops by removing the branch statement within 
         * the j loop. The rewritten code is as follows: 
         *
         * for (i = 0; i < num_rows; i++){
         *      double sum = -A.elements[i * num_cols + i] * ref_x.elements[i];
         *      for (j = 0; j < num_cols; j++)
         *          sum += A.elements[i * num_cols + j] * ref_x.elements[j];
         * }
         *
         * new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
         *
         * I recommend using this code snippet within your GPU kernel implementation.
         * 
         * */

        // Check for convergence and update the unknowns
        ssd = 0.0; // Sum of squared differences 
        for (i = 0; i < num_rows; i++){
            ssd += (new_x.elements[i] - ref_x.elements[i]) * (new_x.elements[i] - ref_x.elements[i]);
            ref_x.elements[i] = new_x.elements[i];
        }
        num_iter++;
        mse = sqrt(ssd); // Mean squared error
        printf ("MSE during teration %d is %f \n", num_iter, mse);
        if (mse <= THRESHOLD)
            done = 1;
    }

    printf ("Convergence achieved after %d iterations \n", num_iter);
    free (new_x.elements);
}
    
/* Function to display statistic related to the Jacobi solution. */
void
display_jacobi_solution(const Matrix A, const Matrix ref_x, const Matrix B)
{
	double diff = 0.0;
	unsigned int num_rows = A.num_rows;
    unsigned int num_cols = A.num_columns;
	
    for(unsigned int i = 0; i < num_rows; i++){
		double line_sum = 0.0;
		for(unsigned int j = 0; j < num_cols; j++){
			line_sum += A.elements[i * num_cols + j] * ref_x.elements[j];
		}
		
        diff += fabsf(line_sum - B.elements[i]);
	}

	printf("Average diff between LHS and RHS: %f \n", diff/(float)num_rows);
}

