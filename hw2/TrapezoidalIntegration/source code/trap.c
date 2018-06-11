/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -o trap trap.c -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * Author: Naga Kandasamy
 * Date modified: 2/11/2018
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
#define NUM_TRAPEZOIDS 100000000
#define NUM_THREADS 16

/**/
struct trapData{
	int id;
	double integral;
	float a;
	int n;
	float h;
};
void *trap (void *);

/**/

double compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);

int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);
	struct timeval start, stop, gstart, gstop;
	gettimeofday(&start, NULL);

	double reference = compute_gold(a, b, n, h);
	printf("Reference solution computed on the CPU = %f \n", reference);

    gettimeofday(&stop, NULL);
    printf("CPU run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Write this function to complete the trapezoidal rule using pthreads. */
	
	gettimeofday(&gstart, NULL);
	double pthread_result = compute_using_pthreads(a, b, n, h);
	printf("Solution computed using pthreads = %f \n", pthread_result);
	gettimeofday(&gstop, NULL);
	printf("Pthread run time = %0.2f s. \n", (float)(gstop.tv_sec - gstart.tv_sec + (gstop.tv_usec - gstart.tv_usec)/(float)1000000));
} 


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 * Output: (x+1)/sqrt(x*x + x + 1)

 */
float f(float x) {
		  return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double compute_gold(float a, float b, int n, float h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  

/* Complete this function to perform the trapezoidal rule using pthreads. */
double compute_using_pthreads(float a, float b, int n, float h)
{/**/
	int i,j,k;
	pthread_t threads[NUM_THREADS];
	double integral;
	integral=(f(a)+f(b))/2.0;
	struct trapData* traps = malloc(NUM_THREADS*sizeof(struct trapData));

	for(i=0;i<NUM_THREADS;i++){
		traps[i].id=i;
		traps[i].integral=0;
		traps[i].a=a;
		traps[i].n=n;
		traps[i].h=h;
		pthread_create(&threads[i], NULL, trap, (void *)&traps[i]);
	}

	for(j=0;j<NUM_THREADS;j++){
		pthread_join(threads[j], NULL);
	}

	for(k=0;k<NUM_THREADS;k++){
		integral+=traps[k].integral;
	}
	integral=integral*h;
	return integral;
	//return 0.0;
	/**/
}
/**/
void *trap(void *s){
	int k;
	struct trapData* threadedTraps=(struct trapData*) s;
	int id=threadedTraps->id;
	double integral=threadedTraps->integral;
	float a=threadedTraps->a;
	int n=threadedTraps->n;
	float h=threadedTraps->h;

	for(k=id;k<=n-1;k+=NUM_THREADS){
		integral+=f(a+k*h);
	}
	threadedTraps->integral=integral;
	pthread_exit(0);
}
/**/
