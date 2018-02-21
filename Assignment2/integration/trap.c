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

#define NUM_RUNS 10 

/* struct to store args for each thread */ 
typedef struct thread_args
{
    int n;
    float a;
    float b;
    float h;
    double sum;
} thread_args;

unsigned long compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);

int num_threads = 2;

int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	// printf("The height of the trapezoid is %f \n", h);

    struct timeval begin, end;

    gettimeofday(&begin, NULL);
	double reference = compute_gold(a, b, n, h);
    gettimeofday(&end, NULL);
    unsigned long serial_time = end.tv_usec-begin.tv_usec+(end.tv_sec-begin.tv_sec)*1000000;
    printf("Single-threaded solution computed in \t%lu us \n", serial_time);

    //printf("Reference solution computed on the CPU = %f \n", reference);

	/* Write this function to complete the trapezoidal rule using pthreads. */
    int run;

    for( ; num_threads <= 16; num_threads *= 2) {
	    printf("\n>>>>> NUM_THREADS = %d <<<<<\n", num_threads);
        
        unsigned long time = 0;
        for(run = 0; run < NUM_RUNS; run++) {
	        time += compute_using_pthreads(a, b, n, h);
        }

        time /= NUM_RUNS;
        double speedup = (double)serial_time / time;
        printf("Average speedup over %d iterations:\t%f\n",
               NUM_RUNS, speedup);
    }
	//printf("Solution computed using pthreads = %f \n", pthread_result);
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

void *calcTrap(void *argstruct)
{
    thread_args *args = (thread_args*)argstruct;
    args->sum = compute_gold(args->a, args->b, args->n, args->h);
    return NULL;
}

/* Complete this function to perform the trapezoidal rule using pthreads. */
unsigned long compute_using_pthreads(float a, float b, int n, float h)
{
    struct timeval begin, end;  /* timing structs */
    double global_sum;          /* sum of all trapezoids */
    float trap_width = (b-a)/num_threads; /* width of each trapezoid */
    
    /* allocate memory for each thread */
    pthread_t* threads = malloc(num_threads*sizeof(pthread_t));

    /* prep arguments for each thread */
    thread_args* args = malloc(num_threads*sizeof(thread_args));

    int i;
    for(i=0; i<num_threads; ++i) {
        args[i].n = n/num_threads;
        args[i].a = a+(trap_width*i);
        args[i].b = a+(trap_width*(i+1));
        args[i].h = h;
    }

    // start timer
    gettimeofday(&begin, NULL);
   
    /* spin up threads */
    for(i=0; i<num_threads; ++i) {
        pthread_create(&threads[i], NULL, calcTrap, &args[i]);
    }

    /* wait for all threads to finish */
    for(i=0; i<num_threads; ++i) {
        pthread_join(threads[i], NULL);
        global_sum+=args[i].sum;
    }

    gettimeofday(&end, NULL);
    unsigned long time = end.tv_usec-begin.tv_usec+(end.tv_sec-begin.tv_sec)*1000000;

    printf("\t\t>> %lu\n", time);
    
    return time;
}
