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
solver_kernel_optimized()
{
//copy src and dist to shared 

//update the values 

//calculate diff (think this may stay in global) 

}

#endif /* _MATRIXMUL_KERNEL_H_ */
