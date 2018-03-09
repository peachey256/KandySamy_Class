#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */

__global__ void vector_dot_product(float* Ad, float* Bd, float* Cd)
{
	int k=blockDim.x*gridDim.x; 
	int tid=threadIdx.x+(blockDim.x+blockIdx.x); 
    extern __shared__ double C_shared[];
	int num_strides = *(n_c)/k; 
	if (*(n_c)%k>0)
		num_strides++; 
	C_shared[tid]=0; // C is the size the number of threads 
	int i; 
	for(i=0; i<num_strides; i++)
		if(tid+(k*i)<*(n_c))
			C_shared[tid]+=((double)Ad[tid+(k*i)]*(double)Bd[tid+(k*i)]); 
	 /*Now everything is multiplied and loaded into share d memory that is the
 	* size of k number of threads, and reduction needs to be applied to get the
 	* answer*/
	__syncthreads();
	int depth=1; 
	int stride; 
	for(i=1; i<k; i*=2){
		stride=k/(2*depth); 
		if(tid<stride)
			C_shared[tid]+=C_shared[tid+stride]; 
		depth++; 
		__syncthreads();	
	}
	*Cd=(float)C_shared[0]; //copy back to global memory

}




#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
