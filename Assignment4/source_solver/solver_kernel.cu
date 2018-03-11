#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

__global__ void 
solver_kernel_naive(float *src, float *dest, double *diff)
{
//update the values for each thread, probabaly will need a stride.
    int blockID      = blockIdx.y  * gridDim.x  + blockIdx.x;
    int threadOffset = threadIdx.y * blockDim.x + threadIdx.x;
    int threadId     = blockID*(blockDim.y * blockDim.x) + threadOffset;

    //calculate diff
}

__global__ void 
solver_kernel_optimized(){
//copy src and dist to shared 

//update the values 

//calculate diff (think this may stay in global) 

}

#endif /* _MATRIXMUL_KERNEL_H_ */
