 /* Device code. */
#include "gauss_eliminate.h"

__global__ void gauss_division_kernel(float *A, int k)
{
	int tid=k+1+(blockDim.x*blockIdx.x+threadIdx.x); 
	int number_of_updates = MATRIX_SIZE-k-1; //dont do anything before k. 	
	int n_threads=GRID_SIZE*BLOCK_SIZE; 
	int num_strides=number_of_updates/(n_threads); 
	if(number_of_updates%(n_threads))
		num_strides++;
	int stride;  
	for(stride=0; stride<num_strides; stride+=n_threads)
		if((tid+stride)<MATRIX_SIZE)
			A[k*MATRIX_SIZE+(tid+stride)]/=A[k*MATRIX_SIZE+k]; 
	//at this point all the elements in row k after col k are divided by the
	//value at row k, col k. still have to set k=1... 
}

__global__ void gauss_eliminate_kernel(float *A, int k)
{
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    // figure out if we need striding
    int n_threads   = blockDim.x * gridDim.x;
    int num_strides = MATRIX_SIZE / n_threads;
    if(MATRIX_SIZE % n_threads)
        num_strides++;

    if (idxX == 0  && idxY == 0)
        A[k * MATRIX_SIZE + k] = 1.0f;

    // stride in X and Y directions
	for( ; idxY < MATRIX_SIZE; idxY+=n_threads ) {
	    for( ; idxX < MATRIX_SIZE; idxX+=n_threads ) {
            
            // TODO: A[i,j] = A[i,j] - A[i,k] * A[k,j];
            A[idxY*MATRIX_SIZE + (idxX)] = 
                (A[idxY*MATRIX_SIZE + idxX]
                - A[idxY*MATRIX_SIZE + k])
                * A[k*MATRIX_SIZE + idxX];
        }

	    A[idxY * MATRIX_SIZE + k] = 0.0f; 
        __syncthreads();
    }
}

