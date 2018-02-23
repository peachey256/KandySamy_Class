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
	int thread_id_i = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_id_j = blockIdx.y * blockDim.y + threadIdx.y; 
	int stride = blockDim.x * gridDim.x; 
	// stride should be the same in the x and y direction because everything
	// is square. 
	
	int thread_id_j_org = thread_id_j; 
	while(thread_id_i < MATRIX_SIZE) { 
		y[thread_id_i]=0; //initialize
		while(thread_id_j<MATRIX_SIZE){
			Yd[thread_id_i]+=Ad[thread_id_i*MATRIX_SIZE+thread_id_j]*Xd[thread_id_j]; 
			thread_id_j+=stride; 
		} 
		thread_id_j=thread_id_j_org;
		thread_id_i+=stride;
	}
}  


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
