#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

__global__ void 
solver_kernel_naive(float *src, float *dest, double *diff)
{
    //update the values for each thread, probabaly will need a stride.
    int blockID      = blockIdx.y  * gridDim.x  + blockIdx.x;
    int blockOffset  = blockID*(blockDim.y * blockDim.x);

    int stride = gridDim.x * blockDim.x;
    int gridWidth = blockDim.x * gridDim.x;

    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    for( ty++ ; ty < GRID_DIMENSION; ty += stride) {
        for( tx++ ; tx < GRID_DIMENSION; tx += stride) {
            
            float tmp = dest[ty * gridWidth + tx];

            dest[ty * gridWidth + tx] = 
                0.2*(src[ty * gridWidth + tx]+
                     src[(ty+1) * gridWidth + tx]+
                     src[(ty-1) * gridWidth + tx]+
                     src[ty * gridWidth + (tx+1)]+
                     src[ty * gridWidth + (tx-1)]);

            //calculate diff and add to total diff
            double newDiff = fabs(dest[ty*gridWidth + tx] - tmp);
            atomicAdd(diff, newDiff);
        }
    }
}

__global__ void 
solver_kernel_optimized(float *src, float *dest, double *diff)
{
    // create some shared memory
    __shared__ float _src_shared[BLOCK_SIZE * BLOCK_SIZE];

    // find yo self in global
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    // find yo self in current block (for shared mem)
    int threadLoc = threadIdx.y * blockDim.x + threadIdx.x;
    
    // width of matrix
    int gridWidth = blockDim.x  * gridDim.x;

    // subtract 2 from stride b/c of border
    int stride = gridDim.x * blockDim.x - 2;

    
    // stride stride stride
    for( ; ty < GRID_DIMENSION; ty += stride) {
        for( ; tx < GRID_DIMENSION; tx += stride) {
           
	    // copy src and dest to shared
            _src_shared[threadLoc] =  src[ty * gridWidth + tx];

            // gotta wait for the slowpokes 
            __syncthreads();
            
            // don't run if we're on the border
            // should we also ignore the first column & row??
            if ( threadIdx.x && threadIdx.y && 
		 (threadIdx.x < blockDim.x-1 && threadIdx.y < blockDim.y-1) &&
		 (ty<(GRID_DIMENSION-1) && tx<(GRID_DIMENSION-1))) {

                 double newDest = 
                    (float)0.2*(_src_shared[threadIdx.y * blockDim.x + threadIdx.x]+
                         _src_shared[(threadIdx.y+1) * blockDim.x + threadIdx.x]+
                         _src_shared[(threadIdx.y-1) * blockDim.x + threadIdx.x]+
                         _src_shared[threadIdx.y     * blockDim.x + (threadIdx.x+1)]+
                         _src_shared[threadIdx.y     * blockDim.x + (threadIdx.x-1)]);
                
           	double tmp = (double)dest[ty*gridWidth + tx];
           	dest[ty*gridWidth + tx] = (float)newDest;
                
                // calculate diff and add to total diff
                //  ... diff stays global
            	double newDiff = fabs(newDest - tmp);
            	atomicAdd(diff, newDiff);
            }

            // copy dst_shared back to dest
            // dest[ty * gridWidth + tx] = diff;
        }
    }
}

#endif /* _MATRIXMUL_KERNEL_H_ */
