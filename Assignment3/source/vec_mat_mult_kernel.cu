#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

//double atomicAdd (double* address, double val);

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
	int thread_id= blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockDim.y*gridDim.y; 
	int j; 
	float y_temp=0;  //initialize
	for(j=0; j<n; j++)
		y_temp+=Ad[thread_id*n+j]*Xd[j]; 
	Yd[thread_id]=y_temp; 
}  


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
    // allocate some shared memory
    __shared__ double M_shared[TILE_SIZE][TILE_SIZE];
    __shared__ double N_shared[TILE_SIZE]; 
	
    // locate yo self within tile - TB and tile size are the same. 
    const unsigned int tileCol = threadIdx.x;
    const unsigned int tileRow = threadIdx.y;

    // locate yo self within Array
    //  ... only true for first file
    const unsigned int row = blockDim.x * blockIdx.x + tileRow;
    //const unsigned int col = blockDim.x * blockIdx.x + tileCol;

    // number of tiles we're going to need
    // ... add an extra if not evently divisble
    double partSum = 0.0f;
    int temp;
    int k;

    // moves tile across matrix
    for(k=0; k<MATRIX_SIZE; k+=TILE_SIZE) {
        // check M edge conditions for this tile
        if(k + tileCol < MATRIX_SIZE && row < MATRIX_SIZE)
            M_shared[tileRow][tileCol] = Ad[row*MATRIX_SIZE + k + tileCol];
        else
            M_shared[tileRow][tileCol] = 0.0f;

        if (k + tileCol < MATRIX_SIZE) 
            N_shared[tileCol] = Xd[k+tileCol];
        else
            N_shared[tileCol] = 0.0f;

        __syncthreads();

        for(temp = 0; temp < TILE_SIZE; temp++)
            partSum += M_shared[tileRow][temp] * N_shared[temp];

        __syncthreads();
    }

    if (row < MATRIX_SIZE)
        Yd[row] = (float)partSum;

}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
