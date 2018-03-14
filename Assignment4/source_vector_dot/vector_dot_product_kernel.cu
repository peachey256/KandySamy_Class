#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */
__constant__ int n_c[1]; // allocation on the kernel

__global__ void vector_dot_product(float* Ad, float* Bd, float* Cd)
{
	int k=THREAD_COUNT; //blockDim.x*gridDim.x; 
	int tid=threadIdx.x+(blockDim.x*blockIdx.x); 
    __shared__ double C_shared[THREAD_COUNT];
    int n_c_local = n_c[0];
	int num_strides = n_c_local/k; 
	if (n_c[0]%k>0)
		num_strides++; 
    if(tid < n_c_local)
	    C_shared[tid]=0; // C is the size the number of threads 
	int i; 
	for(i=0; i<num_strides; i++)
		if(tid<THREAD_COUNT)
			if((tid+(k*i))<n_c_local)
				C_shared[tid]+=((double)Ad[tid+(k*i)]*(double)Bd[tid+(k*i)]); 

	 /*Now every thing is multiplied and loaded into share d memory that is the
 	* size of k number of threads, and reduction needs to be applied to get the
 	* answer*/  
	__syncthreads(); 
	int stride; 
	for(stride=k; stride>0; stride/=2){
		if(tid<stride && tid+stride < k)
			C_shared[tid]+=C_shared[tid+stride]; 
		__syncthreads();	
	}
	if (tid==0)
		*Cd=(float)C_shared[0]; //copy back to global memory

}




#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
