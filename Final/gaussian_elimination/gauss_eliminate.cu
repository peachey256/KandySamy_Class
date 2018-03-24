/* 
   Host-side code for Gaussian elimination. 

    Author: Naga Kandasamy
    Date modified: 3/10/2018
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#include "gauss_eliminate_kernel.cu"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

extern "C" int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void gauss_eliminate_on_device(const Matrix M, Matrix P);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
void write_matrix_to_file(const Matrix M);
float get_random_number(int, int);
void checkCUDAError(const char *msg);
int checkResults(float *reference, float *gpu_result, int num_elements, float threshold);
void writeToFile(float *A);
void zero_out_lower_cpu(float *A);

float CPU_time, GPU_time;

int 
main(int argc, char** argv) 
{
    // Initialize the random number generator with a seed value
	srand(time(NULL));

    // Matrices for the program
	Matrix  A; // The NxN input matrix
	Matrix  U; // The upper triangular matrix 
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
    
    struct timeval startCPU, stopCPU;
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 

	// Perform Gaussian elimination on the CPU 
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);


    gettimeofday(&startCPU, NULL); 
	int status = compute_gold(reference.elements, A.elements, A.num_rows);
    gettimeofday(&stopCPU, NULL); 
	CPU_time=stopCPU.tv_sec-startCPU.tv_sec+(stopCPU.tv_usec-startCPU.tv_usec)/(float)1000000; 
    printf("CPU took %0.3f\n", CPU_time);
	
	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination on the CPU was successful. \n");

	// Perform the vector-matrix multiplication on the GPU. Return the result in U
	gauss_eliminate_on_device(A, U);
    
	// check if the device result is equivalent to the expected solution
    int num_elements = MATRIX_SIZE*MATRIX_SIZE;
    int tol = 0.001f;
    int res = checkResults(reference.elements,U.elements,num_elements,tol);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    float speedup = CPU_time / GPU_time;
    printf("\n >> Speed Up = %f\n", speedup);

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


void 
gauss_eliminate_on_device(const Matrix A, Matrix U)
{
    struct timeval startGPU, stopGPU; 
    Matrix A_on_device, U_on_device; 
		
    //allocate memory on GPU 
    A_on_device=allocate_matrix_on_gpu(A);
    U_on_device=allocate_matrix_on_gpu(U);

    //copy memory to GPU 
    copy_matrix_to_device(A_on_device,A);

    //make the thread blocks and grid jawn 
    dim3 grid(GRID_SIZE); 
    dim3 thread_block(BLOCK_SIZE); 

    // sizes get assigned later
    dim3 elim_grid;
    dim3 elim_tb;

    int k; 
    int num_blocks, num_cols;

    gettimeofday(&startGPU, NULL);
    for(k=0; k<MATRIX_SIZE; k++) {

        // launch division for current k
	gauss_division_kernel<<<grid, thread_block>>>(A_on_device.elements, k);
	cudaThreadSynchronize(); 

        // launch eliminate kernel for k
	int num_blocks = ceil((MATRIX_SIZE-1) - k / BLOCK_SIZE);
	elim_grid = dim3(num_blocks<GRID_SIZE?num_blocks:GRID_SIZE);
	elim_tb   = dim3(BLOCK_SIZE);
        gauss_eliminate_kernel2<<<elim_grid, elim_tb>>>(A_on_device.elements, k);
        
	// zero out subdiagonal elements in column k
        zero_out_column<<<20, 1024>>>(A_on_device.elements, k);
        cudaThreadSynchronize();
    }
    gettimeofday(&stopGPU, NULL);
    GPU_time=stopGPU.tv_sec-startGPU.tv_sec+(stopGPU.tv_usec-startGPU.tv_usec)/(float)1000000; 
    printf("GPU took %0.3f\n", GPU_time);


    copy_matrix_from_device(U, A_on_device); 
    U.elements[MATRIX_SIZE*MATRIX_SIZE-1] = 1.0f;

    //free all the GPU memory 
    cudaFree(A_on_device.elements); 
}

// Allocate a device matrix of same size as M.
Matrix 
allocate_matrix_on_gpu(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}
// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix 
allocate_matrix(int num_rows, int num_columns, int init)
{

    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void 
copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void 
copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void 
print_matrix(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float 
get_random_number(int min, int max)
{
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

// Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1
int 
perform_simple_check(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
        if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;
	
    return 1;
} 

void zero_out_lower_cpu(float *A) 
{
    for(int idx = 1; idx <= (MATRIX_SIZE*(MATRIX_SIZE-1)/2); idx++) {
        int rvLinear = (MATRIX_SIZE*(MATRIX_SIZE-1))/2-idx;
	    int k = floor( (sqrtf(1+8*rvLinear)-1)/2 );
	    int j = rvLinear - k*(k+1)/2;
	    int y = MATRIX_SIZE-j-1;
		int x = MATRIX_SIZE-(k+1)-1;
		A[y*MATRIX_SIZE + x] = 0.0f;
        printf("zeroing: [%d, %d] = %f\n", y, x, A[y*MATRIX_SIZE + x]);
    }
}

// Writes the matrix to a file 
void 
write_matrix_to_file(const Matrix M)
{
	FILE *fp;
	fp = fopen("matrix.txt", "wt");
	for(unsigned int i = 0; i < M.num_rows; i++){
        for(unsigned int j = 0; j < M.num_columns; j++)
            fprintf(fp, "%f", M.elements[i*M.num_rows + j]);
        }
    fclose(fp);
}

void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}

void writeToFile(float* A)
{
    FILE *fp = fopen("diffMatrix.txt", "w+");
    for(int y = 0; y < MATRIX_SIZE; y++) {
    	for(int x = 0; x < MATRIX_SIZE; x++) {
	        fprintf(fp, "%f\t", fabsf(A[y*MATRIX_SIZE + x]));
	    }
	fprintf(fp, "\n");
    }
    fclose(fp);
}

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
 
    float* diff = (float *)malloc(sizeof(float)*MATRIX_SIZE*MATRIX_SIZE);

    //int xDiverge, yDiverge;
    double totalSum = 0.0f, currDiff;

    for(int i = 0; i < num_elements; i++) {
        if(reference[i] == 0.0f && gpu_result[i] == 0.0f)
            currDiff = 0.0f;
        else if(reference[i] == 0.0f) {
            currDiff = fabsf(gpu_result[i]);
        }
        else {
	        currDiff = fabsf((reference[i] - gpu_result[i])/reference[i]);
            if (currDiff > threshold)
                checkMark = 0;
        }
        
        totalSum += currDiff;
        diff[i] = currDiff;
        if (currDiff > epsilon)
            epsilon = currDiff;
    }

    printf("Total Diff  = %f\n", totalSum);
    printf("Max epsilon = %f. \n", epsilon); 
    writeToFile(diff);

    return checkMark;
}
