#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */

__global__ void vector_dot_product(float* Ad, float* Bd, float* Cd)
{
	k=blockDim.x*gridDim.x; 
	tid=threadIdx.x+(blockDim.x+blockIdx.x); 
   __shared__ double C_shared[k];
	num_strides = n_c/k; 
	if (n%k>0)
		num_strides++; 
	C_shared[tid]=0; // C is the size the number of threads 
	int i; 
	for(i=0; i<num_strides; i++)
		if(tid+(k*i)<n)
			C_shared[tid]+=(a[tid+(k*i)]*b[tid+(k*i)]; 
	/*Now everything is multiplied and loaded into shared memory that is the
 	* size of k number of threads, and reduction needs to be applied to get the
 	* answer*/
	__syncthreads();
	depth=1; 
	int i; 
	for(i=0, 2**i<k, i++){
		stride=k/(2*depth); 
		if(tid<stride)
			C_shared[tid]+=C_shared[tid+stride]; 
		depth++; 
		__syncthreads();	
	}
	if(tid==0)
		Cd=C_shared[tid]; //copy back to global memory

}




#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
