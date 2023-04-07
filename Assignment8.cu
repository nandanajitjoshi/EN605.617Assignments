/*Matrix A - define in CSR, convert to BSR*/
/*Vector b, vector r, Vector r-o*/
/*rho, alphs, pmega, itr - Scalars*/
//rho, i-1 and rho_i, omega_i-1 and omega_i
//Steps 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include "helper_functions.h"  // helper for shared functions common to CUDA Samples
#include "helper_cuda.h"       // helper function CUDA error checking and initialization
/*Initialize CSR*/

/*gernTriDiag generates a tridiogonal system of matrices in tridiagonal format
input I is the row vector, input J is the col vector, and val holds the values
Matrix generated is - [1 0 0.....0
                      -1 2 -1....0
                       0 -1 2 -1..0
                       .
                       .
                       .
                       0..-1 2 -1
                       0.....0...1]

Matrix is indexed in column-1 format
*/
void genTridiag(int *I, int *J, double*val, int rows)
{


    I[0] = 1;
    int start = 0;   //Holds the index for the vector in CSR//

    for (int i = 1; i <= rows; i++)
    {
        start = (i-1)*3;
        
        if ( i == 1){
            /*1st row*/

            /*Col representation = [1 2 3]*/
            J[start] = 1;
            J[start + 1] = 2;
            J[start+2] = 3;
            
            /*Values = [1 0 0]*/
            val[start] = 1;
            val[start +1] = 0;
            val[start+2] = 0;
            
            
            }else if ((i== rows)) {
                /*Last row*/
                //start = 1 + (i-1)*3;
                J[start ] = rows-2;
                J[start +1 ] = rows -1;
                J[start + 2] = rows;
              
                /*Values = [0 0 1]*/
                val[start] = 0;
                val[start+1] = 0;
                val[start + 2] = 1;
            
            } else {
                //start = 1 + (i-1)*3;
                /*Intermediate row*/
                J[start ] = i-1;
                J[start +1 ] = i;
                J[start + 2] = i+1;

                /*Values = [-1 2 -1]*/
                val[start ] = -1;
                val[start +1 ] = i;
                val[start + 2] = -1;
            }

          I[i] = 3 + I[i-1]; 
    }
}

/*Generates the RHS vector for linear solve
* All values are set to 1
*/
void genRHS (double* RHS, int rows){
    for (int i=0; i < rows; i++){
        RHS[i] = 1.0;
    }
}

/* Prints the results of the sparse solve 
* First prints the tridiagonal coefficent matrix
* Followed by the RHS and results*/

void printOutput(int *I, int *J, double*val, double* RHS, double* Soln, int rows){
    int valIndex = 0;
    int offset = 0; 

    printf ("The coefficient matrix is : \n"); 
    printf ("\n");
     for (int i = 1; i <= rows; i++){
        for (int j=1; j<= rows; j++){
            if (j== J[valIndex]){


                if (offset < 3){
                    printf ("%1.0f ", val[valIndex]);
                    valIndex++;
                    offset++;
                }

            } else {
                printf("0 "); 

            }


            }
            //valIndex ++;
            offset = 0;
            printf("\n");

        }

    printf ("\n The RHS vector is : \n");

    for (int i=0; i<rows; i++){
        printf ("%f\n", RHS[i]);
    }

    printf ("\n The solution is:\n");

    for (int i = 0; i < rows ; i++){
        
        printf ("%f\n", Soln[i]);       

    }
       
}

/* Solves a matrix system Ax=b
* Uses cusparse library
* A is represented in a CSR format
* I represent the row indices and J represents col indices
*/

float TriDiagSolve(int rows){

    // int *I, *J; 
    // double *val_host ; 
    // double* RHS = NULL ; 
    // double* Soln = NULL; 

    int * I_dvc, *J_dvc; 
    double* val_dvc = NULL; 
    double *RHS_dvc = NULL; 
    double *Soln_Dvc = NULL; 

   

    //int rows = 5; 
    int dimBlock = 1;   //CSR = BSR wth block dimension 1

    /*Allocate memory for host-side arrays*/
    int* I = (int *)malloc(sizeof(int)*(rows+1));
    int *J = (int *)malloc(sizeof(int)*rows *3);
    double *val_host = (double *)malloc(sizeof(double) * rows *3);
    double *RHS = (double *)malloc(sizeof(double) * rows);
    double *Soln = (double *)malloc(sizeof(double) * rows);
  

    /*Generate the sparse coefficient matrix*/
    genTridiag (I, J, val_host, rows);

    /*Generate RHS vector*/
    genRHS( RHS, rows); 

    /*Allocate memory on the device for the arrays*/
    checkCudaErrors(cudaMalloc((int **)&J_dvc, rows *3*sizeof(int)));
    checkCudaErrors(cudaMalloc((int **)&I_dvc, (rows+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((double **)&val_dvc, rows *3*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&RHS_dvc, (rows)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Soln_Dvc, (rows)*sizeof(double)));

    /*Initialize the variables to be used in the linear solve*/
    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    cusparseMatDescr_t descr_coeff = 0;
    bsrsv2Info_t  info_coeff = 0;
    const cusparseSolvePolicy_t policy_coeff = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW; //Doesn't matter for blockdIM = 1
    int pBufferSize; 
    void* pBuffer = 0;
    double alpha = 1;
    int structural_zero, numerical_zero;    //To check for singularities in the coefficient matrix

    /*1 - Create descriptor for coeff matrix*/
    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSetMatDiagType(descr_coeff, CUSPARSE_DIAG_TYPE_NON_UNIT);

    /*2 - Create info for linear solve*/
    cusparseCreateBsrsv2Info(&info_coeff);


    /*Create timer variables*/
    cudaEvent_t startTime, stopTime; 
    float time; 
    cudaEventCreate (&startTime);
    cudaEventCreate (&stopTime);
    cudaEventRecord (startTime, 0);


    /* 3 - Transfer data from host to device*/
    checkCudaErrors (cudaMemcpy(J_dvc, J,  rows *3*sizeof(int), 
            cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(val_dvc, val_host, rows *3*sizeof(double), 
            cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(I_dvc, I, (rows+1)*sizeof(int),
             cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy( RHS_dvc, RHS,(rows)*sizeof(double), 
            cudaMemcpyHostToDevice ));

    /*4 - Allocate buffer space for linear solve*/
    checkCudaErrors(cusparseDbsrsv2_bufferSize(handle, dir_coeff, trans_coeff, 
            rows,rows*3, descr_coeff, val_dvc,I_dvc, J_dvc, 
            dimBlock, info_coeff, &pBufferSize));
    checkCudaErrors(cudaMalloc((void**)&pBuffer, pBufferSize));

    /*5 - Analyze coefficient matrix and report any singularities */
    checkCudaErrors(cusparseDbsrsv2_analysis(handle, dir_coeff, trans_coeff, 
            rows, rows*3,descr_coeff, val_dvc,I_dvc, J_dvc, 
            dimBlock, info_coeff, policy_coeff, pBuffer));

    cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff, 
            &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        //Singularities found
        printf("U(%d,%d) is zero\n", structural_zero, structural_zero);
    }

    /*6 - Perform the matrix solve*/
    checkCudaErrors(cusparseDbsrsv2_solve(handle, dir_coeff, trans_coeff, rows, rows*3, &alpha,
               descr_coeff, val_dvc,I_dvc, J_dvc, dimBlock, info_coeff, RHS_dvc, Soln_Dvc,
               policy_coeff, pBuffer));

    cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff, 
            &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        /*0 found in a pivot column - invalid solution*/
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    /* 7 - Transfer result data back to host*/
    checkCudaErrors (cudaMemcpy( Soln, Soln_Dvc,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    /*Record end time*/
    cudaEventRecord (stopTime, 0);
    cudaEventSynchronize (stopTime);
    cudaEventElapsedTime (&time, startTime, stopTime);
    cudaEventDestroy (startTime);
    cudaEventDestroy (stopTime); 

    //Print output
    printOutput (I, J, val_host, RHS,Soln, rows);

    //Free CUDA memory
    cudaFree (I_dvc);
    cudaFree (J_dvc);
    cudaFree (val_dvc);
    cudaFree (RHS_dvc);
    cudaFree (Soln_Dvc);  

    return time;

}

/*Initializes input vectors X and Y for xuBlas
*/
void initializeVectors(double* X, double* Y, int rows){    

    for (int i=0; i < rows; i++){
        X[i] = (double)rand()/RAND_MAX; 
    }
    
    for (int i=0; i < rows; i++){
        Y[i] = 2*i%rows; 
    }
}

/*Adds two vectors, X and Y as Y = Y+X
* Uses cuBlas library
* Vector Y is overwritten
* Return : Execution time
*/

float vectorAdd(int rows){

    /*Declare the vectors on host*/
    double* X = (double *)malloc(sizeof(double)*(rows));
    double * Y = (double *)malloc(sizeof(double)*rows);
    double alpha = 1; 

    
    initializeVectors (X, Y, rows); 


    // printf ("\n Vector 1 [X] is \n");
    // for (int i=0; i < rows; i++){
    //     printf ("%f\n", X[i] );
    // }
    // printf ("\n Vector 2 [Y] is \n");
    // for (int i=0; i < rows; i++){
    //     printf ("%f\n", Y[i] );
    // }

    /*Declare and allocate vectors on device side*/
    double* dvc_X, *dvc_Y; 
    checkCudaErrors(cudaMalloc((double **)&dvc_X, rows*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&dvc_Y, rows*sizeof(double)));


    /*Initialize cuBlas handles*/
    cublasHandle_t handleBlas = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handleBlas);
    checkCudaErrors (cublasStatus);


    /*Create timer variables*/
    cudaEvent_t startTime, stopTime; 
    float time; 
    cudaEventCreate (&startTime);
    cudaEventCreate (&stopTime);
    cudaEventRecord (startTime, 0);

    
    /*Transfer data to host*/
    checkCudaErrors(cudaMemcpy(dvc_X, X, rows*sizeof(double), 
                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dvc_Y, Y, rows*sizeof(double), 
                cudaMemcpyHostToDevice));



    /*Add vectors - Y = X + Y*/
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,dvc_X, 1,dvc_Y, 1));

    /*Transfer result back to host*/
    checkCudaErrors(cudaMemcpy(Y, dvc_Y, rows*sizeof(double), 
                cudaMemcpyDeviceToHost));

    /*Record end time*/
    cudaEventRecord (stopTime, 0);
    cudaEventSynchronize (stopTime);
    cudaEventElapsedTime (&time, startTime, stopTime);
    cudaEventDestroy (startTime);
    cudaEventDestroy (stopTime); 


    // printf ("\n Output is \n");
    // for (int i=0; i < rows; i++){
    //     printf ("%f\n", Y[i] );
    // }

    /*Free up memory*/

    cudaFree (X);
    cudaFree(Y); 
   
   return time;


    
}

/* Starting point of the program
* Creates a linear system of a problem size from cmd prompts
* Solves the system using cuSparse
* Adds two vectors of problemSize using cuBlas
* 
*/

int main(int argc, char** argv){
    float time; 
    int rows = 5;

    if (argc >= 2) {
            rows = atoi(argv[1]);
    }

    // validate command line arguments
    if (rows< 3) {
        
        rows = 3;
        printf("Warning: Problem size can't be less than 3\n");
        printf("The total number of threads will be modified  to 3\n");
    }

    printf ("\n\nSolving  linear  equations using cuSparse library\n"); 
    time = TriDiagSolve( rows);
    printf ("The time taken for linear solve is \n");
    printf ("%3.1f mus", time );

    printf ("\n\nAdding two vectors using cuBlas library\n");
    time = vectorAdd (rows);  
    printf ("The time taken for vector add is \n");
    printf ("%3.1f mus", time );
}
