#include "jacobi_iteration.h"

// Write the GPU kernel to solve the Jacobi iterations
__global__ void jacobi_iteration_kernel (float * Ad, float * Bd, float * xd, float * diff)
{
	    // allocate some shared memory
    __shared__ double A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ double x_shared[TILE_SIZE]; 
	
    // locate yo self within tile - TB and tile size are the same. 
    const unsigned int tileCol = threadIdx.x;
    const unsigned int tileRow = threadIdx.y;

    // locate yo self within Array
    //  ... only true for first file
    const unsigned int row = blockDim.x * blockIdx.x + tileRow;
    const unsigned int col = blockDim.x * blockIdx.x + tileCol;

    // number of tiles we're going to need
    // ... add an extra if not evently divisble
    double partSum = Bd[col];
    int temp;
    int k;

    // moves tile across matrix
    for(k=0; k<MATRIX_SIZE; k+=TILE_SIZE) {
        // check M edge conditions for this tile
        if(k + tileCol < MATRIX_SIZE && row < MATRIX_SIZE)
            A_shared[tileRow][tileCol] = Ad[row*MATRIX_SIZE + k + tileCol];
        else
            A_shared[tileRow][tileCol] = 0.0f;

        if (k + tileCol < MATRIX_SIZE)
            x_shared[tileCol] = Xd[k+tileCol];
        else
            x_shared[tileCol] = 0.0f;

        __syncthreads();

			//NEED to implement with reduction later. 
        for(temp = 0; temp < TILE_SIZE; temp++)
				if(temp!=tileRow)//subtract out all the non diagonal elements 
            	partSum -= A_shared[tileRow][temp] * X_shared[temp];

        __syncthreads();
    }

    if (row < MATRIX_SIZE){
		*diff = (Xd[row]-partSum)*( Xd[row]-partSum); 	
    	Xd[row] = (float)partSum/A_shared[tileRow][tileRow];
		//everything is divided by the diagonal element 
	}



}



