 /* Device code. */
#include "gauss_eliminate.h"

__global__ void gauss_division_kernel(double *A, int k)
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

__global__ void gauss_eliminate_kernel(double *A, int k)
{
    int idxX = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y + k + 1;

    // figure out if we need striding
    int n_threads   = blockDim.x * gridDim.x;
/*    int num_strides = MATRIX_SIZE / n_threads;
    if(MATRIX_SIZE % n_threads)
        num_strides++;*/

    if ( !(idxX-k-1) && !(idxY-k-1) ) {
        A[k * MATRIX_SIZE + k] = 1.0f;
        //if (k == MATRIX_SIZE-2) {
        //    A[MATRIX_SIZE*MATRIX_SIZE-1] = 1.0f;
        //}
    }

    // stride in X and Y directions
	for( ; idxY < MATRIX_SIZE; idxY+=n_threads ) {
	    for( ; idxX < MATRIX_SIZE; idxX+=n_threads ) {
            
            // TODO: A[i,j] = A[i,j] - A[i,k] * A[k,j];
            double tmp = (double)A[idxY*MATRIX_SIZE+idxX] -
                (double)A[idxY*MATRIX_SIZE + k]*(double)A[k*MATRIX_SIZE + idxX];

            A[idxY*MATRIX_SIZE+idxX] = tmp; 
        }
           
        // zero out element below diagonal after sync
        //A[idxY*MATRIX_SIZE + k] = 0.0f;
    }
}

__global__ void zero_out_lower_kernel(double *A)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int n_threads = blockDim.x * gridDim.x;

    for(int idx=tid+1; idx<=(MATRIX_SIZE*(MATRIX_SIZE-1))/2; idx+=n_threads ) {
        /*tmp = (MATRIX_SIZE*(MATRIX_SIZE-1))/2-idx;
        x = MATRIX_SIZE - floor((sqrtf(1+8*tmp)-1)/2);
        y = MATRIX_SIZE - tmp - k*(k+1)/2 - 1;
        A[y*MATRIX_SIZE + x] = 100.0f;*/
		
	  	int rvLinear = (MATRIX_SIZE*(MATRIX_SIZE-1))/2-idx;
	    int k = floor( (sqrtf(1+8*rvLinear)-1)/2 );
	    int j = rvLinear - k*(k+1)/2;
	    int y = MATRIX_SIZE-j-1;
		int x = MATRIX_SIZE-(k+1)-1;
		A[y*MATRIX_SIZE + x] = 0.0L;
	}
}

__global__ void float_to_double(float *A, double *B)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    int n_threads = blockDim.x * gridDim.x;

    for (int y = tidy; y < MATRIX_SIZE; y += n_threads) {
        for (int x = tidx; x < MATRIX_SIZE; x += n_threads) {
            B[y*MATRIX_SIZE + x] = (double)A[y*MATRIX_SIZE + x];
        }
    }
}

__global__ void double_to_float(float *A, double *B)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    int n_threads = blockDim.x * gridDim.x;

    for (int y = tidy; y < MATRIX_SIZE; y += n_threads) {
        for (int x = tidx; x < MATRIX_SIZE; x += n_threads) {
            A[y*MATRIX_SIZE + x] = (float)B[y*MATRIX_SIZE + x];
        }
    }
}
