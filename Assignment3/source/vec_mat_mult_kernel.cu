/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

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
	//Multiply A and X
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
