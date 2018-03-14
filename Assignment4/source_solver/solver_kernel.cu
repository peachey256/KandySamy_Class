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
            //float tmp = dest[blockOffset+ty*blockDim.x+tx];

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
    __shared__ float _dst_shared[BLOCK_SIZE * BLOCK_SIZE];

    // find yo self in global
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    // find yo self in current block (for shared mem)
    int threadLoc = threadIdx.y * blockDim.x + threadIdx.x;
    int gridWidth = blockDim.x  * gridDim.x;

    // subtract 2 from stride b/c of border
    int stride = gridDim.x * blockDim.x - 2;

    // stride stride stride
    for( ; ty < GRID_DIMENSION - stride; ty += stride) {
        for( ; tx < GRID_DIMENSION - stride; tx += stride) {
           
            // copy src and dest to shared
            _src_shared[threadLoc] =  src[ty * gridWidth + tx];
            _dst_shared[threadLoc] = dest[ty * gridWidth + tx];

            // gotta wait for the slowpokes 
            __syncthreads();
            
            // don't run if we're on the border
            if ( tx && ty && tx<(blockDim.x) && ty<(blockDim.y) ) {
                float tmp = _dst_shared[threadLoc];

                _dst_shared[threadIdx.y * blockDim.x + threadIdx.x] = 
                    0.2*(_src_shared[threadIdx.y     * blockDim.x + threadIdx.x]+
                         _src_shared[(threadIdx.y+1) * blockDim.x + threadIdx.x]+
                         _src_shared[(threadIdx.y-1) * blockDim.x + threadIdx.x]+
                         _src_shared[threadIdx.y     * blockDim.x + (threadIdx.x+1)]+
                         _src_shared[threadIdx.y     * blockDim.x + (threadIdx.x-1)]);

                // calculate diff and add to total diff
                //  ... diff stays global
                // this is going to introduce some thread divergence
                double newDiff = fabs(_dst_shared[threadLoc] - tmp);
                atomicAdd(diff, newDiff);
            }

            // copy dst_shared back to dest
            dest[ty * gridWidth + tx] = _dst_shared[threadLoc];
        }
    }
 
//update the values 

//calculate diff (think this may stay in global) 

}

#endif /* _MATRIXMUL_KERNEL_H_ */
