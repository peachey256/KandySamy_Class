 /* Device code. */
#include "gauss_eliminate.h"

__global__ void gauss_division_kernel( float *A, int k)
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

__global__ void gauss_eliminate_kernel2( float *A, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int stride_len = blockDim.x * gridDim.x;

    if(!(col-k-1))
        A[k*MATRIX_SIZE + k] = 1.0f;

    for ( ; col<MATRIX_SIZE-1; col+=stride_len)
        for (int row=k+1; row<MATRIX_SIZE-1; row++)
            A[row*MATRIX_SIZE+col]-=A[k*MATRIX_SIZE+col]*A[row*MATRIX_SIZE+k];
}

__global__ void gauss_eliminate_kernel( float *A, int k)
{
    int idxX = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y + k + 1;

    int n_threads   = blockDim.x * gridDim.x;

    if ( !(idxX-k-1) && !(idxY-k-1) )
        A[k * MATRIX_SIZE + k] = 1.0f;

    // stride in X and Y directions
	for( ; idxX < MATRIX_SIZE; idxX+=n_threads )
        A[idxY*MATRIX_SIZE+idxX] -= (double)A[idxY*MATRIX_SIZE + k]*(double)A[k*MATRIX_SIZE + idxX];
        
        __syncthreads();
}

__global__ void zero_out_column(float *A, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int n_threads = blockDim.x * gridDim.x;

    for (int row = tid + k + 1; row < MATRIX_SIZE; row += n_threads)
        A[row*MATRIX_SIZE + k] = 0;
}

__global__ void zero_out_lower_kernel(double *A)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int n_threads = blockDim.x * gridDim.x;

    for(int idx=tid+1; idx<=(MATRIX_SIZE*(MATRIX_SIZE-1))/2; idx+=n_threads ) {
	  	int rvLinear = (MATRIX_SIZE*(MATRIX_SIZE-1))/2-idx;
	    int k = floor( (sqrtf(1+8*rvLinear)-1)/2 );
	    int j = rvLinear - k*(k+1)/2;
	    int y = MATRIX_SIZE-j-1;
		int x = MATRIX_SIZE-(k+1)-1;
		A[y*MATRIX_SIZE + x] = (float)0;
	}
}

__global__ void float_to_double(float *A, double *B)
{
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int n_threads = blockDim.x * gridDim.x;

    for (int y = tidy; y < MATRIX_SIZE; y += n_threads) {
        for (int x = tidx; x < MATRIX_SIZE; x += n_threads) {
            B[y*MATRIX_SIZE + x] = (double)A[y*MATRIX_SIZE + x];
        }
    }
}

__global__ void double_to_float(float *A, double *B)
{
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int n_threads = blockDim.x * gridDim.x;

    for (int y = tidy; y < MATRIX_SIZE; y += n_threads) {
        for (int x = tidx; x < MATRIX_SIZE; x += n_threads) {
            A[y*MATRIX_SIZE + x] = (float)B[y*MATRIX_SIZE + x];
        }
    }
}
