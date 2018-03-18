#include "jacobi_iteration.h"

// Write the GPU kernel to solve the Jacobi iterations
__global__ void jacobi_iteration_kernel (float * Ad, float * Bd, float * Xd, float * x_new, double * diff)
{
	    // allocate some shared memory
    __shared__ double A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ double x_shared[TILE_SIZE]; 
	__shared__ double part_red[TILE_SIZE][TILE_SIZE]; 	
    // locate yo self within tile - TB and tile size are the same. 
    int tileCol = threadIdx.x;
    int tileRow = threadIdx.y;

    // locate yo self within Array
    //  ... only true for first file
    int row = blockDim.y * blockIdx.y + tileRow;
    int col = blockDim.x * blockIdx.x + tileCol;

    // number of tiles we're going to need
    // ... add an extra if not evently divisble
    int k;
	part_red[tileRow][tileCol]=0; 
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

		//all the tiles are reduced into one tile 
		if(k+tileCol!=row)
			part_red[tileRow][tileCol]+=(double)A_shared[tileRow][tileCol]*(double)x_shared[tileCol];
		else 
			part_red[tileRow][tileCol]+=0; //dont add up the diagonal element 
        
		__syncthreads();
    }
	//reduce that tile into a column 
	int stride; 
	for(stride=TILE_SIZE/2; stride>0; stride/=2){
		if(tileCol<stride && tileCol+stride < TILE_SIZE)
			part_red[tileRow][tileCol]+=part_red[tileRow][tileCol+stride]; 
		__syncthreads();	
	}

    if (col==0){ 
		x_new[row] = ((double)Bd[row]-part_red[tileRow][0])/(double)Ad[row*MATRIX_SIZE+row]; 
		double error =(Xd[row]-x_new[row])*(Xd[row]-x_new[row]); 
		atomicAdd(diff, error); 
		//everything is divided by the diagonal element 
	}
 	

}



