 /* Device code. */
#include "gauss_eliminate.h"

__global__ void gauss_division_kernel(float *A, int k)
{
	int tid=k+1+(blockDim.x*blockIdx.x+threadIdx.x); 
	int number_of_updates = MATRIX_SIZE-k-1; //dont do anything before k. 	
	int n_threads=GRID_SIZE*BLOCK_SIZE; 
	int num_stride=number_of_updates/(n_threads); 
	if(number_of_updates%(n_threads))
		num_strides++;
	int stride;  
	for(stride=0; stride<num_strides; strides+=n_threads)
		if((tid+stride)<MATRIX_SIZE)
			A[k*MATRIX_SIZE+(tid+stride)]/=A[k*MATRIX_SIZE+k]; 
	//at this point all the elements in row k after col k are divided by the
	//value at row k, col k. still have to set k=1... 
}



__global__ void gauss_eliminate_kernel(float *A, int k)
{
	
}

