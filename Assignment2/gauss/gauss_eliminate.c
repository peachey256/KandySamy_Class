/* Gaussian elimination code.
 * Author: Naga Kandasamy
 * Date created: 02/07/2014
 * Date of last update: 2/11/2018
 * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -lpthread -std=c99 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int num_rows, int num_columns, int init);
void gauss_eliminate_using_pthreads (Matrix);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);


int
main (int argc, char **argv)
{
    /* Check command line arguments. */
    if (argc > 1){
        printf ("Error. This program accepts no arguments. \n");
        exit (0);
    }

    /* Matrices for the program. */
    Matrix A;			    // The input matrix
    Matrix U_reference;		// The upper triangular matrix computed by the reference code
    Matrix U_mt;			// The upper triangular matrix computed by the pthread code

    /* Initialize the random number generator with a seed value. */
    srand (time (NULL));

  
    /* Allocate memory and initialize the matrices. */
    A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	            // Allocate and populate a random square matrix
    U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the reference result
    U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	        // Allocate space for the multi-threaded result

    /* Copy the contents of the A matrix into the U matrices. */
    for (int i = 0; i < A.num_rows; i++){
        for (int j = 0; j < A.num_rows; j++){
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    printf ("Performing gaussian elimination using the reference code. \n");
    struct timeval start, stop;
    gettimeofday (&start, NULL);
    
    int status = compute_gold (U_reference.elements, A.num_rows);
  
    gettimeofday (&stop, NULL);
    printf ("CPU run time = %0.2f s. \n",
            (float) (stop.tv_sec - start.tv_sec +
                (stop.tv_usec - start.tv_usec) / (float) 1000000));

  
    if (status == 0){
        printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
        exit (0);
    }

  
    status = perform_simple_check (U_reference);	// Check that the principal diagonal elements are 1 
    if (status == 0){
        printf ("The upper triangular matrix is incorrect. Exiting. \n");
        exit (0);
    }
    printf ("Single-threaded Gaussian elimination was successful. \n");

  
    /* Perform the Gaussian elimination using pthreads. The resulting upper 
     * triangular matrix should be returned in U_mt */
    gauss_eliminate_using_pthreads (U_mt);

  
    /* check if the pthread result matches the reference solution within a specified tolerance. */
    int size = MATRIX_SIZE * MATRIX_SIZE;
    int res = check_results (U_reference.elements, U_mt.elements, size, 0.0001f);
    printf ("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

  
    /* Free memory allocated for the matrices. */
    free (A.elements);
    free (U_reference.elements);
    free (U_mt.elements);

    return 0;
}

void *parallel_gold(void* Matrices_ptr){
	/*Get the stuff from the struct*/
	TwoMat Matrices = *((TwoMat *)Matrices_ptr); 
	float * U = Matrices.U; //U.elements
	float * temp = Matrices.temp; //temp.elements
	int k = Matrices.k; 
	int tid=pthread_self();
  /* do the section of the for loop here */ 
	/* gonna needs to make a copy everytime or figure out how to pass it in. */ 
	
	int n = MATRIX_SIZE/Matrices.num_threads; 
	/*All the examples are evenly divisiable*/ 
	for(i=tid*n; i<(tid+1)*n; i++)
		for(j=0; j<MATRIX_SIZE; j++){
			if(i==k)
			{//Division 
				if (U[n *k + k] == 0){
					printf("Numerical instability detected. The principal
							diagonal element is zero. \n");
					k = j = n;
				}
				temp[n * k + j] = (float)(U[n * k + j] / U[n * k + k]);
			}else{
			//Elimination
			   temp[n * i + j] = U[n * i + j] -\
					(U[n * i + k] * (float)(U[n * k + j] /U[n * k + k]));
			}
	}
}


/* Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (Matrix U)
{
	
	/* malloc the threads */
	pthread_t* thread_handles; 
	thread_count=4;
 	thread_handles=malloc(thread_count*sizeof(pthread_t));
	/*make the structure to pass to the threads*/
	temp = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	
	TwoMat Matrices; 
	Matrices.U=U.elements; 
	Matrices.temp=temp.elements;
  	Matrices.num_threads=thread_count; 
	float * fuck; 	
	for(k=0; k<MATRIX_SIZE; k++){
		/*Spawn of threads into a reference function*/
		for(thread =0; thread < thread_count; thread++) 
			pthread_create(&thread_handles[thread], NULL, parallel_gold, (void *)&Matrices); 
		
		/*Join the threads*/
		for(thread=0; thread<thread_count; thread++)
			pthread_join(thread_handles[thread], NULL); 
	
		
		for(j=k; j<MATRIX_SIZE; j++){
			Matrices.U[MATRIX_SIZE*k+j]=Matrices.temp[MATRIX_SIZE*k+j];
			Matrices.U[MATRIX_SIZE*j+k]=Matrices.temp[MATRIX_SIZE*j+k];	
		}
		fuck = Matrices.U;  
		Matrices.U=Matrices.temp; 
		Matrices.temp=fuck;  
	}
	free(temp); 
}


/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
    for (int i = 0; i < size; i++)
        if (fabsf (A[i] - B[i]) > tolerance)
            return 0;
    return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *) malloc (size * sizeof (float));
  
    for (unsigned int i = 0; i < size; i++){
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
    return (float)
        floor ((double)
                (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
    for (unsigned int i = 0; i < M.num_rows; i++)
        if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.0001)
            return 0;
  
    return 1;
}
