/*  Purpose: Calculate definite integral using the trapezoidal rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Author: Naga Kandasamy, Michael Lui
 * Date: 6/22/2016
 *
 * Compile: gcc -o trap trap.c -lpthread -lm
 * Usage:   ./trap
 *
 *
 * Note:    The function f(x) is hardwired.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <pthread.h>

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
#define NUM_TRAPEZOIDS 100000000
#define NUM_THREADS 16 /* Number of threads to run. */


/*------------------------------------------------------------------
 * Function:    func
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 * Output: (x+1)/sqrt(x*x + x + 1)
 */
__attribute__((const)) float func(float x) 
{
	return (x + 1)/sqrt(x*x + x + 1);
}  


double compute_gold(float, float, int, float (*)(float));
double compute_using_pthreads(float, float, int, float (*)(float));

int main(int argc, char *argv[])
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	
	struct timeval begin,end;

	gettimeofday(&begin,NULL);
	double reference = compute_gold(a, b, n, func);
	gettimeofday(&end,NULL);
//	printf("%lu \t %lu \n",begin.tv_usec,end.tv_usec);
	printf("Reference solution computed in %luus \n", end.tv_usec-begin.tv_usec+(end.tv_sec-begin.tv_sec)*1000000);
	printf("Reference solution computed on the CPU = %f \n", reference);
	
	gettimeofday(&begin,NULL);
	double pthread_result = compute_using_pthreads(a, b, n, func); /* Write this function using pthreads. */
	gettimeofday(&end,NULL);
	printf("Solution computed using pthreads = %f \n", pthread_result);
} 

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, f
 * Return val:  Estimate of the integral 
 */
double compute_gold(float a, float b, int n, float(*f)(float))
{
	float h = (b-a)/(float)n; /* 'Height' of each trapezoid. */

	double integral = (f(a) + f(b))/2.0;
	
	for (int k = 1; k <= n-1; k++) 
		integral += f(a+k*h);
	
	integral = integral*h;
	
	return integral;
}  

typedef struct numbers
{
	float a; 
	float b; 
	int n; 
	float(*f)(float);  
	double sum; 
} numbers; 


void * someFunction(void *argstruct)
{
	numbers* param=(numbers*)argstruct; 
	param->sum = compute_gold(param->a, param->b, param->n, param->f); 	
	return NULL; 
}

double compute_using_pthreads(float a, float b, int n, float(*f)(float))
{
	int i;
	double sum;
	float range=(b-a)/(float)NUM_THREADS;
	struct timeval begin,end;
	numbers* arglist=malloc(sizeof(numbers)*NUM_THREADS);
	pthread_t* threads=malloc(NUM_THREADS*sizeof(pthread_t));

	for(i=0;i<NUM_THREADS;++i){
		arglist[i].n=n/NUM_THREADS;
		arglist[i].a=a+(range*i);
		arglist[i].b=a+(range*(i+1));
		arglist[i].f=f;
	}
	gettimeofday(&begin,NULL);
	for(i=0;i<NUM_THREADS;++i)
		pthread_create(&threads[i], NULL, someFunction, &arglist[i]); 	
	for(i=0;i<NUM_THREADS;++i){
		pthread_join(threads[i], NULL); 
		sum+=arglist[i].sum;
	}
	gettimeofday(&end,NULL);
//	printf("%lu \t %lu \n",begin.tv_usec,end.tv_usec);
	printf("Pthread solution computed in %luus \n", end.tv_usec-begin.tv_usec+(end.tv_sec-begin.tv_sec)*1000000);
	return sum;
}  
