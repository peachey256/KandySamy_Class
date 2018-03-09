#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

__global__ void 
solver_kernel_naive(GRID_STRUCT *src, GRID_STRUCT *dest, double diff)
//update the values for each thread, probabaly will need a stride. 

//calculate diff

}

__global__ void 
solver_kernel_optimized(){
//copy src and dist to shared 

//update the values 

//calculate diff (think this may stay in global) 

}

#endif /* _MATRIXMUL_KERNEL_H_ */
