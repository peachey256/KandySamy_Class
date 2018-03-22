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


int 
main(int argc, char** argv) 
{
    // Matrices for the program
	Matrix  A; // The NxN input matrix
	Matrix  U; // The upper triangular matrix 
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
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

	int status = compute_gold(reference.elements, A.elements, A.num_rows);
	
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
    int res = checkResults(reference.elements, U.elements, num_elements, 0.001f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


void 
gauss_eliminate_on_device(const Matrix A, Matrix U)
{
	Matrix A_on_device, U_on_device; 
		
	//allocate memory on GPU 
	A_on_device=allocate_matrix_on_gpu(A);
    U_on_device=allocate_matrix_on_gpu(U);

    // allocate matrix of double precision elements
    /*double* A_double;
    int size = A.num_rows * A.num_columns * sizeof(double);
    cudaMalloc((void**)&A_double, size);*/

	//copy memory to GPU 
	copy_matrix_to_device(A_on_device,A);

    dim3 cpGrid(GRID_MAX, GRID_MAX);
    dim3 cpTB(BLOCK_MAX, BLOCK_MAX);

    //float_to_double<<<cpGrid, cpTB>>>(A_on_device.elements, A_double);
	
	//make the thread blocks and grid jawn 
	dim3 grid(GRID_SIZE); 
	dim3 thread_block(BLOCK_SIZE); 

    // sizes get assigned later
    dim3 elim_grid;
    dim3 elim_tb;

	int k; 
	//for all the k 
	for(k=0; k<MATRIX_SIZE-1; k++){
		//They need to be launched this way to ensure that synchronization
		//happens between all thread blocks 

		//launch division for that k_i
		gauss_division_kernel<<<grid, thread_block>>>(A_on_device.elements, k);
		cudaThreadSynchronize(); 

        // calculate how large of a threadblock/ grid we need
        int currDim = MATRIX_SIZE - (k + 1);

      
        if (currDim <= BLOCK_MAX) {
            elim_tb = dim3(currDim, currDim);
            elim_grid = dim3(1, 1);
        } 
        
        else if ( currDim < GRID_MAX * BLOCK_MAX ) {
            elim_tb = dim3(BLOCK_MAX, BLOCK_MAX);

            // grid = # of times 32 goes into BLOCK_MAX * GRID_MAX / 
            int tmpSize = (int)floor(currDim / BLOCK_MAX) + (currDim % BLOCK_MAX ? 1 : 0);

            elim_grid = dim3(tmpSize, tmpSize);
        }

        else {
            elim_tb = dim3(BLOCK_MAX, BLOCK_MAX);
            elim_grid = dim3(GRID_MAX, GRID_MAX);
        }

        printf(">> k = %d, grid = %dx%d, block = %dx%d\n",
                k, elim_tb.x, elim_tb.y, elim_grid.x, elim_grid.y);

		//launch elimination for that k_i
		gauss_eliminate_kernel<<<elim_grid, elim_tb>>>(A_on_device.elements, k); 
		cudaThreadSynchronize(); 
	}

    int threadsNeeded = (MATRIX_SIZE * (MATRIX_SIZE - 1)) / 2;
    dim3 zero_grid;
    dim3 zero_tb;

    if ( threadsNeeded < BLOCK_MAX * BLOCK_MAX ) {
        zero_tb   = dim3(threadsNeeded);
        zero_grid = dim3(1);
    } else if (threadsNeeded < BLOCK_MAX * BLOCK_MAX * GRID_MAX * GRID_MAX) {
        zero_tb   = dim3(BLOCK_MAX * BLOCK_MAX);

        int gridSize = floor(threadsNeeded / (BLOCK_MAX * BLOCK_MAX));
        if (threadsNeeded % (BLOCK_MAX * BLOCK_MAX)) gridSize++;
        zero_grid = dim3(gridSize);

    } else {
        zero_tb   = dim3(BLOCK_MAX * BLOCK_MAX);
        zero_grid = dim3(GRID_MAX * GRID_MAX);
    }

    zero_tb   = dim3(16);
    zero_grid = dim3(4);

    printf("zero_grid = %d\n", zero_grid.x);
    printf("zero_tb   = %d\n", zero_tb.x);

    zero_out_lower_kernel<<<zero_grid, zero_tb>>>(A.elements);
    cudaThreadSynchronize();

    //double_to_float<<<cpGrid, cpTB>>>(A_on_device.elements, A_double);
    //cudaThreadSynchronize();

	//copy memory back to CPU 
	copy_matrix_from_device(U, A_on_device); 
    U.elements[MATRIX_SIZE*MATRIX_SIZE-1] = 1.0f;

	//free all the GPU memory 
    // cudaFree(A_double);
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
 
    printf("\n\n_______REFERENCE_______\n");
    for(int y = 0; y < 8; y++) {
        for(int x = MATRIX_SIZE-8; x < MATRIX_SIZE; x++) {

    //for(int y = 0; y < 5; y++) {
    //    for(int x = 0; x < 5; x++) {

            printf("%f\t", reference[y*MATRIX_SIZE+x]);
        }
        printf("\n");
    }

    printf("\n________RESULT________\n");
    for(int y = 0; y < 8; y++) {
        for(int x = MATRIX_SIZE-8; x < MATRIX_SIZE; x++) {

    //for(int y = 0; y < 5; y++) {
    //    for(int x = 0; x < 5; x++) {
            printf("%f\t", gpu_result[y*MATRIX_SIZE+x]);
        }
        printf("\n");
    }
    printf("\n");

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
