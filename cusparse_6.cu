/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. AN use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
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

const char *sSDKname     = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, double*val, int M, int N)
{

    I[0] = 1;
    int start = 0; 

    for (int i = 1; i <= M*N; i++)
    {
        start = (i-1)*5;
        
        if ( i <= N){
            J[start + 4] = i+N*(M-1);
            J[start] = i-1;
            J[start+1] = i;
            J[start+2] = i+1;
            J[start + 3] = i+N;

            val[start + 4] = 0;
            val[start] = 0;
            val[start+1] = 1;
            val[start+2] = 0;
            val[start + 3] = 0;


            if (i%N ==1){
            J[start + 4] = i+N*(M-1);
            J[start] = i;
            J[start+1] = i+1;
            J[start+2] = i+N-1;
            J[start + 3] = i+N;

            val[start + 4] = 0;
            val[start] = 1;
            val[start+1] = 0;
            val[start+2] = 0;
            val[start + 3] = 0;
            
            }else if ((i%N==0)) {
                J[start + 4] = i+N*(M-1);
                J[start] = i-N+1;
                J[start+1] = i-1;
                J[start+2] = i;
                J[start + 3] = i+N;

                val[start + 3] = 0;
                val[start+4] = 0;
                val[start] = 0;
                val[start+1] = 0;
                val[start + 2] = 1;
            
            }
            
        } else if (i > (M-1)*N){

            J[start] = i-(M-1)*N;
            J[start+1] = i-N;
            J[start+2] = i-1;
            J[start+3] = i;
            J[start + 4] = i+1;

            val[start + 4] = 0;
            val[start] = 0;
            val[start+1] = 0;
            val[start+3] = 1;
            val[start + 2] = 0;

            if (i%N ==1 ){
                J[start] = i-(M-1)*N;
                J[start+1] = i-N;
                J[start+2] =i;
                J[start+3] = i+1;
                J[start + 4] = i+N-1;

                val[start + 4] = 0;
                val[start] = 0;
                val[start+1] = 0;
                val[start+3] = 0;
                val[start + 2] = 1;
            } else if (i%N==0){
                J[start] = i-(M-1)*N;
                J[start+1] = i-N;
                J[start+2] =i-N+1;
                J[start+3] = i-1;
                J[start + 4] = i;

                val[start + 4] = 1;
                val[start] = 0;
                val[start+1] = 0;
                val[start+3] = 0;
                val[start + 2] = 0;
            
            }
            


        } else if ( i%N == 1 ){
                J[start + 3] = i+N-1;
                J[start+4] = i+N;
                J[start] = i-N;
                J[start+1] = i;
                J[start + 2] = i+1;

                val[start + 3] = 0;
                val[start+4] = 0;
                val[start] = 0;
                val[start+1] = 1;
                val[start + 2] = 0;

        } else if ((i%N==0)){
                J[start + 3] = i;
                J[start+4] = i+N;
                J[start] = i-N;
                J[start+1] = i-N+1;
                J[start + 2] = i-1;

                val[start + 3] = 1;
                val[start+4] = 0;
                val[start] = 0;
                val[start+1] = 0;
                val[start + 2] = 0;
        } else {
            J[start + 4] = i+N;
            J[start] = i-N;
            J[start+1] = i-1;
            J[start+2] = i;
            J[start + 3] = i+1;


            val[start + 4] = 1;
            val[start] = 1;
            val[start+1] = 1;
            val[start+2] = -4;
            val[start + 3] = 1;
            
        }


        


        // start = (i-1)*5;
        // J[start] = i - N;
        // J[start+1] = i-1;
        // J[start+2] = i;
        // J[start+3] = i+1;
        // J[start + 4] = i+N; 

        // val[start] = -0.025;
        // val[start+1] = -0.025;
        // val[start+2] = 6;
        // val[start+3] = -0.025;
        // val[start+4] = -0.025;



        // if ( i > (M-0.025)*N){
        //     J[start + 1] = i-N;
        //     J[start+2] = i-1;
        //     J[start+3] = i;
        //     J[start+4] = i+1;
        //     J[start] = N - (N*M-i);

        //     val[start+3] = 6;
        //     val[start+2] = -0.025;

        //     if (i==M*N){
        //         J[start + 2] = M*N-N+1;
        //         J[start+3] = i-1;
        //         J[start+4] = i;

        //         val[start+4] = 6;
        //         val[start+3] = -0.025;
        //     }
        // }

        // if ((i%N==1)&& (i!=1)){
        //     J[start] = i - N;
        //     J[start+1] = i;
        //     J[start+2] = i+1;
        //     J[start+3] = i+N-1;
        //     J[start + 4] = i+N;

        //     val[start] = -0.025;
        //     val[start+1] = 6;
        //     val[start+2] = -0.025;
        //     val[start+3] = -0.025;
        //     val[start + 4] = -0.025;

        //     if (i==((M*N)-N+1)){

        //         J[start] = 1;
        //         J[start+1] = i-N;
        //         J[start+2] = i;
        //         J[start+3] = i+1;
        //         J[start + 4] = N*M;

        //         val[start] = -0.025;
        //         val[start+1] = -0.025;
        //         val[start+2] = 6;
        //         val[start+3] = -0.025;
        //         val[start + 4] = -0.025;
                
        //     }             
            
        // }

        // if ((i%(N)==0)&& (i!=M*N)){
        //     J[start] = i - N;
        //     J[start+1] = i-N+1;
        //     J[start+2] = i-1;
        //     J[start+3] = i;
        //     J[start + 4] = i+N; 

        //     val[start] = -0.025;
        //     val[start+1] = -0.025;
        //     val[start+2] = -0.025;
        //     val[start+3] = 6;
        //     val[start + 4] = -0.025;


        //     if (i==N){
        //         J[start] = 1;
        //         J[start+1] = i-1;
        //         J[start+2] = i;
        //         J[start+3] = i+N;
        //         J[start + 4] = N*M;

        //         val[start] = -0.025;
        //         val[start+1] = -0.025;
        //         val[start+2] = 6;
        //         val[start+3] = -0.025;
        //         val[start + 4] = -0.025;
        //     }
        // }

        I[i] = 5 + I[i-1]; 

      
    }
    

    


}

int main(int argc, char **argv)
{
    // int M = 0, N = 0, nz = 0, 
    double*val = NULL;
    double* ValItr = NULL; 
    // const double tol = 1e-5f;
    // const int max_iter = 10000;
    double*X;
    double* Y;
    double *Z;
    double*rhs;
    // double a, b, na, r0, r1;
    int *d_col, *d_row;
    double*d_val, *d_x, dot, *d_y, *d_z, *d_p;
    int*db_col, *db_row;
    double*db_val;
    //double*d_r, *d_p, *d_Ax;
    //int k;
    //double alpha, beta, alpham1;
    int *I = NULL, *J = NULL, *ITr = NULL, *JTr = NULL; 
  


    int rows = 4;
    int cols = 4;

    int N=rows*cols;
    int nnz = 5*rows*cols;
    int nnzb = nnz;
   
    double toler = 1e-5;
    float boost = 4;
    const double alpha = 1;
    const double beta = 0;
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    // M = N = 5;
    
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nnz);
    val = (double *)malloc(sizeof(double) * nnz);

    ITr = (int *)malloc(sizeof(int)*(N+1));
    JTr = (int *)malloc(sizeof(int)*45);
    ValItr = (double *)malloc(sizeof(double) * 45);

    genTridiag(I, J, val, rows,cols);
    X = (double*)malloc(sizeof(double)*(N));
    Y = (double*)malloc(sizeof(double)*(N));
    Z = (double*)malloc(sizeof(double)*(N));


    for (int i = 0; i < nnz; i++){
        printf ("%f\t", val[i]);
        if (((i+1)%5)==0){
            printf("\n");
        }   
    }


    
    printf ("\n Columns:\n");

    for (int i = 0; i < nnz; i++){

            printf ("%d\t", J[i]);
            if (((i+1)%5)==0){
                printf("\n");
        }   
        }

    printf ("\n Rows:\n");

    for (int i = 0; i < N+1; i++){
        
        printf ("%d\n", I[i]);       

    }
    for (int i=0; i<N; i++){
        X[i] = 1.0;
    }

    printf ("\n X:\n");

    for (int i = 0; i < N+1; i++){
        
        printf ("%f\n", X[i]);       

    }


    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);


    //define matrix variables
    cusparseMatDescr_t descr_coeff = 0;
    cusparseMatDescr_t descr_Lc = 0;
    cusparseMatDescr_t descr_Uc = 0;

    bsrilu02Info_t info_coeff = 0;
    bsrsv2Info_t  info_coeff_2 = 0;
    bsrsv2Info_t info_coeff_3 = 0;
    // csrsv2Info_t   info_Lc = 0;
    // csrsv2Info_t   info_Uc = 0;
    int pBufferSize_coeff;
    int pBufferSize_Lc;
    int pBufferSize_Uc;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    int blockD = 2;


    const cusparseSolvePolicy_t policy_coeff = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lc = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Uc = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_Lc  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Uc  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;


    checkCudaErrors(cusparseStatus);

    checkCudaErrors(cudaMalloc((int **)&d_col, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((int **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((double **)&d_val, nnz*sizeof(double)));
    checkCudaErrors(cudaMalloc((int **)&db_col, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((int **)&db_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((double **)&db_val, nnz*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_x, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_y, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_z, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(double)));
    // checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_col, J, nnz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nnz*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, X, N*sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_y, Y, N*M*sizeof(double), cudaMemcpyHostToDevice));
    
    cusparseXcsr2bsrNnz(handle, dir_coeff, N, N,
        descr_coeff, d_row, d_col, blockD,
        descr_coeff, db_row, &nnzb);

    cusparseDcsr2bsr(handle, dir_coeff, N, N,
        descr_coeff, d_val, d_row, d_col, blockD,
        descr_coeff, db_val, db_row, db_col);
    
    // step 1: create a descriptor which contains
    // - matrix M is base-0.025
    // - matrix L is base-0.025
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    //checkCudaErrors(cusparseStatus);


    cusparseCreateMatDescr(&descr_Lc);
    cusparseSetMatIndexBase(descr_Lc, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_Lc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_Lc, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_Lc, CUSPARSE_DIAG_TYPE_UNIT);


    cusparseCreateMatDescr(&descr_Uc);
    cusparseSetMatIndexBase(descr_Uc, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_Uc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_Uc, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_Uc, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csric02 and two info's for csrsv2
    checkCudaErrors(cusparseCreateBsrilu02Info(&info_coeff));
    checkCudaErrors(cusparseCreateBsrsv2Info(&info_coeff_2));
    checkCudaErrors(cusparseCreateBsrsv2Info(&info_coeff_3));
    // cusparseCreateCsrsv2Info(&info_Uc);
    checkCudaErrors(cusparseStatus);

    // step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
     checkCudaErrors(cusparseDbsrilu02_bufferSize(handle, dir_coeff,  N, nnz,
         descr_coeff, db_val, db_row, db_col, blockD, info_coeff, &pBufferSize_coeff));

     checkCudaErrors(cusparseDbsrsv2_bufferSize(handle, dir_coeff, trans_Lc,   N, nnz,
         descr_Lc, db_val, db_row, db_col, blockD, info_coeff_2, &pBufferSize_Lc));

     checkCudaErrors(cusparseDbsrsv2_bufferSize(handle, dir_coeff, trans_Uc,  N, nnz,
         descr_Uc, db_val, db_row, db_col, blockD, info_coeff_3, &pBufferSize_Uc));

    //checkCudaErrors(cusparseDbsrsv2_bufferSize(handle, dir_coeff, trans_coeff, N, nnz,
      //       descr_coeff, d_val, d_row, d_col, blockD, info_coeff_3, &pBufferSize_coeff));

    pBufferSize = max(pBufferSize_coeff, max(pBufferSize_Lc, pBufferSize_Uc)); 

    checkCudaErrors(cudaMalloc((void**)&pBuffer, pBufferSize));
    //(checkCudaErrors(cusparseStatus));


    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csric02 and csrsv2 simultaneously.

    checkCudaErrors(cusparseDbsrilu02_analysis(handle, dir_coeff,  N, nnz,
        descr_coeff, db_val, db_row, db_col,blockD, info_coeff, 
        policy_coeff, pBuffer));

    cusparseStatus = cusparseXbsrilu02_zeroPivot(handle, info_coeff, &structural_zero);

    // checkCudaErrors(cusparseDbsrsv2_analysis(handle, dir_coeff, trans_coeff, N, nnz,
    //            descr_coeff, d_val, d_row, d_col, blockD, info_coeff_3, policy_coeff, pBuffer));

    // if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    // printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    // }
    // checkCudaErrors(cusparseStatus);

    //cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff_3, &structural_zero);

    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // step 5: M = L * L'
    checkCudaErrors(cusparseDbsrilu02(handle, dir_coeff, N, nnz,
        descr_coeff, db_val, db_row, db_col,blockD, info_coeff, 
        policy_coeff, pBuffer));

    // checkCudaErrors(cusparseDbsrsv2_solve(handle, dir_coeff, trans_coeff, N, nnz, &alpha,
    //            descr_coeff, d_val, d_row, d_col, blockD, info_coeff_3, d_x, d_y,
    //            policy_coeff, pBuffer));

    // checkCudaErrors(cusparseStatus);

    cusparseStatus = cusparseXbsrilu02_zeroPivot(handle, info_coeff, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }


    checkCudaErrors(cusparseDbsrsv2_analysis(handle, dir_coeff, trans_Lc, N, nnz,
                descr_Lc, db_val, db_row, db_col, blockD, info_coeff_2, policy_Lc, pBuffer));
    
    checkCudaErrors(cusparseDbsrsv2_analysis(handle, dir_coeff, trans_Uc, N, nnz,
            descr_Uc, db_val, db_row, db_col, blockD, info_coeff_3, policy_Uc, pBuffer));

    checkCudaErrors(cusparseDbsrsv2_solve(handle, dir_coeff, trans_Lc, N, nnz, &alpha,
               descr_Lc, db_val, db_row, db_col, blockD, info_coeff_2, d_x, d_y,
               policy_Lc, pBuffer));

    cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff_2, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }


    checkCudaErrors(cusparseDbsrsv2_solve(handle, dir_coeff, trans_Uc, N, nnz, &alpha,
               descr_Uc, db_val, db_row, db_col, blockD, info_coeff_3, d_y, d_z,
               policy_Uc, pBuffer));


    cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff_3, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }


    checkCudaErrors(cudaMemcpy( J, d_col,nnz*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( I, d_row,(N+1)*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( val, d_val, nnz*sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( X, d_x, N*sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( Y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( Z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost));

    printf ("\n Matrix:\n");

    for (int i = 0; i < nnz; i++){
        printf ("%f\t", val[i]);
        if (((i+1)%5)==0){
            printf("\n");
        }   
    }


    
    printf ("\n Columns:\n");

    for (int i = 0; i < nnz; i++){

            printf ("%d\t", J[i]);
            if (((i+1)%5)==0){
                printf("\n");
            }
        }

    printf ("\n Rows:\n");

    for (int i = 0; i < N+1; i++){
        
        printf ("%d\n", I[i]); 


    }

    printf ("\n X:\n");

    for (int i = 0; i < N+1; i++){
        
        printf ("%d\n", I[i]);   
        if (((i+1)%5)==0){
            printf("\n");
        }    

    }
    
    printf ("\n Answer:\n");

    for (int i = 0; i < N; i++){
        
        printf ("%f\n", Y[i]);       

    }

    printf ("\n Answer:\n");

    for (int i = 0; i < N; i++){
        
        printf ("%f\n", Z[i]);       

    }

    checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_Uc, N,N, nnz, &alpha,
               descr_Lc, d_val, d_row, d_col, blockD,  d_z, &beta, d_p));


    // cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    // cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
       
    // k = 1;

    // while (r1 > tol*tol && k <= max_iter)
    // {
    //     if (k > 1)
    //     {
    //         b = r1 / r0;
    //         cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
    //         cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
    //     }
    //     else
    //     {
    //         cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
    //     }

    //     cusparseSbsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
    //     cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
    //     a = r1 / dot;

    //     cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
    //     na = -a;
    //     cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

    //     r0 = r1;
    //     cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    //     cudaThreadSynchronize();
    //     printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    //     k++;
    // }

    // cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);

    // doublersum, diff, err = 0.0;

    // for (int i = 0; i < N; i++)
    // {
    //     rsum = 0.0;

    //     for (int j = I[i]; j < I[i+1]; j++)
    //     {
    //         rsum += val[j]*x[J[j]];
    //     }

    //     diff = fabs(rsum - rhs[i]);

    //     if (diff > err)
    //     {
    //         err = diff;
    //     }
    // }

    // cusparseDestroy(cusparseHandle);
    // cublasDestroy(cublasHandle);

    // free(I);
    // free(J);
    // free(val);
    // free(x);
    // free(rhs);
    // cudaFree(d_col);
    // cudaFree(d_row);
    // cudaFree(d_val);
    // cudaFree(d_x);
    // cudaFree(d_r);
    // cudaFree(d_p);
    // cudaFree(d_Ax);

    // // cudaDeviceReset causes the driver to clean up all state. While
    // // not mandatory in normal operation, it is good practice.  It is also
    // // needed to ensure correct operation when the application is being
    // // profiled. Calling cudaDeviceReset causes all profile data to be
    // // flushed before the application exits
    // cudaDeviceReset();

     //printf("Test Summary:  Error amount = %f\n", err);
    // exit((k <= max_iter) ? 0 : 1);
}
