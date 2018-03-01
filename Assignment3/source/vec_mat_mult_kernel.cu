/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

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

    // locate yo self within tile
    const unsigned int tileCol = blockIdx.x;
    const unsigned int tileRow = blockIdx.y;

    // locate yo self within Array
    const unsigned int row = blockDim.y * blockIdx.y + tileRow;
    const unsigned int col = blockDim.x * blockIdx.x + tileCol;

    // number of tiles we're going to need
    // ... add an extra if not evently divisble
    unsigned int numTiles = MATRIX_SIZE / TILE_SIZE;
    if ( MATRIX_SIZE % TILE_SIZE ) numTiles++;

    double partSum = 0.0f;

    int tileNum;
    for (tileNum=0; tileNum < numTiles; tileNum++)
    {
        // read elements of this tile into shared memory
        if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
            M_shared[tileRow][tileCol] = Ad[row*MATRIX_SIZE+col];
            N_shared[tileRow] = Xd[col];
        } else {
            M_shared[tileRow][tileCol] = 0.0f;
            N_shared[tileRow] = 0.0f;
        }

        // wait for all threads to finish populating shared memory
        __syncthreads();

        // mulitply yourself by corresponding element in N
        atomicAdd(&partSum, M_shared[tileRow][tileCol] * N_shared[tileCol]);

        // wait for threads to finish multiplying
        __syncthreads();
    }

    if (col < TILE_SIZE && row < TILE_SIZE)
        Yd[row*TILE_SIZE + col] = (float)partSum;
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
