/* 
Code for the equation solver. 
Author: Naga Kandasamy 
Date modified: 3/4/2018 
*/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h" // This file defines the grid data structure

// Include the kernel code during the preprocessing step
#include "solver_kernel.cu"

extern "C" void compute_gold(GRID_STRUCT *);


/* This function prints the grid on the screen */
void 
display_grid(GRID_STRUCT *my_grid)
{
	for(int i = 0; i < my_grid->dimension; i++)
		for(int j = 0; j < my_grid->dimension; j++)
			printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
		printf("\n");
}


/* This function prints out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
		// Print statistics for the CPU grid
		float min = INFINITY;
		float max = 0.0;
		double sum = 0.0; 
		for(int i = 0; i < my_grid->dimension; i++){
			for(int j = 0; j < my_grid->dimension; j++){
				sum += my_grid->element[i * my_grid->dimension + j]; // Compute the sum
				if(my_grid->element[i * my_grid->dimension + j] > max) max = my_grid->element[i * my_grid->dimension + j]; // Determine max
				if(my_grid->element[i * my_grid->dimension + j] < min) min = my_grid->element[i * my_grid->dimension + j]; // Determine min
				 
			}
		}

	printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}

/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2)
{
    double diff;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff/num_elements);
}

/* This function creates a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_for_cpu, GRID_STRUCT *grid_for_gpu)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_for_cpu->dimension, grid_for_cpu->dimension);
	grid_for_cpu->element = (float *)malloc(sizeof(float) * grid_for_cpu->num_elements);
	grid_for_gpu->element = (float *)malloc(sizeof(float) * grid_for_gpu->num_elements);


	srand((unsigned)time(NULL)); // Seed the the random number generator 
	
	float val;
	for(int i = 0; i < grid_for_cpu->dimension; i++)
		for(int j = 0; j < grid_for_cpu->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE; // Obtain a random value
			grid_for_cpu->element[i * grid_for_cpu->dimension + j] = val; 	
			grid_for_gpu->element[i * grid_for_gpu->dimension + j] = val; 				
		}
}


/* Edit this function skeleton to solve the equation on the device. Store the results back in the my_grid->element data structure for comparison with the CPU result. */
void 
compute_on_device(GRID_STRUCT *src)
{
	//GRID_STRUCT *temp = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));
	GRID_STRUCT *dest = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT));
	dest->dimension = GRID_DIMENSION;
	dest->dimension = src->dimension *src->dimension; 
	
    double diff = 0;
    double *diffPtr = &diff;

    double *Diff_on_device;
    float  *A_on_device;
    float  *B_on_device;
    float  *tmpPtr;
	
    //allocate memory for src, dest, and diff on GPU 
    cudaMalloc((void**)&A_on_device, GRID_DIMENSION*GRID_DIMENSION*sizeof(float));
    cudaMalloc((void**)&B_on_device, GRID_DIMENSION*GRID_DIMENSION*sizeof(float));
    cudaMalloc((void**)&Diff_on_device, sizeof(double));
   
    cudaMemcpy(A_on_device, src->element, GRID_DIMENSION*GRID_DIMENSION*sizeof(float), cudaMemcpyHostToDevice);

    // setup grid and thread blocks
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 thread_block(BLOCK_SIZE, BLOCK_SIZE);
 
    printf("Creating grid of size %dx%d blocks\n", GRID_SIZE, GRID_SIZE);
    printf("Creating block of size %dx%d threads\n", 
            BLOCK_SIZE, BLOCK_SIZE);

	int done = 0, cnt = 0;
	while(!done){

        //launch the kernel
        diff = (double)0;
        cudaMemcpy(Diff_on_device, &diff, sizeof(double), cudaMemcpyHostToDevice);
        printf("executing kernel...\n");
        solver_kernel_naive<<<grid, thread_block>>>(A_on_device, B_on_device,
                Diff_on_device);

        cudaThreadSynchronize();

        //copy diff from the GPU only a single value 
        cudaMemcpy(&diff, Diff_on_device, sizeof(double), cudaMemcpyDeviceToHost);

        printf("GPU iteration %d : diff = %f\n", ++cnt, diff);

        if( (diff/(GRID_DIMENSION*GRID_DIMENSION)) < TOLERANCE ) {
            done = 1;
        }

        // most boring game of ping-pong I've ever played
        tmpPtr = A_on_device;
        A_on_device = B_on_device;
        B_on_device = tmpPtr;
	}
	
    //Copy dest from the GPU, because we need the final output 
    cudaMemcpy(dest->element, A_on_device, GRID_DIMENSION*GRID_DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
	src=dest;
	
    //free memory on GPU 
    cudaFree(A_on_device);
    cudaFree(B_on_device);
    cudaFree(Diff_on_device);
}

/* The main function */
int 
main(int argc, char **argv)
{	 
	/* Generate the grid */
	GRID_STRUCT *grid_for_cpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure

	grid_for_cpu->dimension = GRID_DIMENSION;
	grid_for_cpu->num_elements = grid_for_cpu->dimension * grid_for_cpu->dimension;
	grid_for_gpu->dimension = GRID_DIMENSION;
	grid_for_gpu->num_elements = grid_for_gpu->dimension * grid_for_gpu->dimension;

 	create_grids(grid_for_cpu, grid_for_gpu); // Create the grids and populate them with the same set of random values
	
	printf("Using the cpu to solve the grid. \n");
	compute_gold(grid_for_cpu);  // Use CPU to solve 
	
	// Use the GPU to solve the equation
	compute_on_device(grid_for_gpu);
	
	// Print key statistics for the converged values
	printf("CPU: \n");
	print_statistics(grid_for_cpu);

	printf("GPU: \n");
	print_statistics(grid_for_gpu);
	
    /* Compute grid differences. */
    compute_grid_differences(grid_for_cpu, grid_for_gpu);

	free((void *)grid_for_cpu->element);	
	free((void *)grid_for_cpu); // Free the grid data structure 
	
	free((void *)grid_for_gpu->element);	
	free((void *)grid_for_gpu); // Free the grid data structure 

	exit(0);
}
