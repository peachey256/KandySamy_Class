#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

__global__ void 
solver_kernel_naive(float *src, float *dest, double *diff)
{
    //update the values for each thread, probabaly will need a stride.
    int blockID      = blockIdx.y  * gridDim.x  + blockIdx.x;
    int blockOffset  = blockID*(blockDim.y * blockDim.x);

    // offset by one b/c of edge buffer
    int ty = threadIdx.y + 1;
    int tx = threadIdx.x + 1;

    float tmp = dest[blockOffset+ty*blockDim.x+tx];

    dest[blockOffset+ty*blockDim.x+tx] = 
        0.2*(src[blockOffset+ (ty)  *blockDim.x+(tx)  ]+
             src[blockOffset+ (ty-1)*blockDim.x+(tx)  ]+
             src[blockOffset+ (ty+1)*blockDim.x+(tx)  ]+
             src[blockOffset+ (ty)  *blockDim.x+(tx-1)]+
             src[blockOffset+ (ty)  *blockDim.x+(tx+1)]);

    //calculate diff and add to total diff
    double newDiff = *diff + dest[blockOffset+ty*blockDim.x+tx] - tmp;
    if(newDiff < 0) 
        newDiff *= -1;
    atomicAdd(diff, newDiff);
}

__global__ void 
solver_kernel_optimized()
{
//copy src and dist to shared 

//update the values 

//calculate diff (think this may stay in global) 

}

#endif /* _MATRIXMUL_KERNEL_H_ */
