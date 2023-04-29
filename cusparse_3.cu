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
        J[start] = i - N;
        J[start+1] = i-1;
        J[start+2] = i;
        J[start+3] = i+1;
        J[start + 4] = i+N; 

        val[start] = -1;
        val[start+1] = -1;
        val[start+2] = 4;
        val[start+3] = -1;
        val[start+4] = -1;


        if ( i < N){
            J[start + 4] = M*N-N+i;
            J[start] = i-1;
            J[start+1] = i;
            J[start+2] = i+1;
            J[start + 3] = i+N;

            val[start+1] = 4;
            val[start+2] = -1;


            if (i==1){
                J[start] = i;
                J[start+1] = i+1;
                J[start + 2] = i+N-1;

                val[start] = 4;
                val[start+1] = -1;
            }
        }

        if ( i > (M-1)*N){
            J[start + 1] = i-N;
            J[start+2] = i-1;
            J[start+3] = i;
            J[start+4] = i+1;
            J[start] = N - (N*M-i);

            val[start+3] = 4;
            val[start+2] = -1;

            if (i==M*N){
                J[start + 2] = M*N-N+1;
                J[start+3] = i-1;
                J[start+4] = i;

                val[start+4] = 4;
                val[start+3] = -1;
            }
        }

        if ((i%N==1)&& (i!=1)){
            J[start] = i - N;
            J[start+1] = i;
            J[start+2] = i+1;
            J[start+3] = i+N-1;
            J[start + 4] = i+N;

            val[start+1] = 4;
            val[start+2] = -1;

            if (i==((M*N)-N+1)){

                J[start] = 1;
                J[start+1] = i-N;
                J[start+2] = i;
                J[start+3] = i+1;
                J[start + 4] = N*M;
                
            }    


            
            
        }

        if ((i%(N)==0)&& (i!=M*N)){
            J[start] = i - N;
            J[start+1] = i-N+1;
            J[start+2] = i-1;
            J[start+3] = i;
            J[start + 4] = i+N; 

            val[start+3] = 4;
            val[start+2] = -1;


            if (i==N){
                J[start] = 1;
                J[start+1] = i-1;
                J[start+2] = i;
                J[start+3] = i+N;
                J[start + 4] = N*M;
            }
        }

        I[i] = 5 + I[i-1]; 

      
    }


}

int main(int argc, char **argv)
{
    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    double*val = NULL;
    int *d_col, *d_row;
    double*d_val, *d_x, dot;
    double*d_r, *d_p, *d_Ax;
    int k;
    double alpha, beta, alpham1;

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
    M = N = 4;
    
    I = (int *)malloc(sizeof(int)*(M+1));
    J = (int *)malloc(sizeof(int)*6);
    val = (double*)malloc(sizeof(double)*6);
    //genTridiag(I, J, val, N,M);

    // printf ("\n Matrix:\n");

    // for (int i = 0; i < N*M; i++){
    //     for (int j = 0; j< 5; j++){
    //         printf ("%f\t", val[i*5+j]);
    //     }

    //     printf("\n");
    // }

    
    // printf ("\n Columns:\n");

    // for (int i = 0; i < N*M; i++){
    //     for (int j = 0; j< 5; j++){
    //         printf ("%d\t", J[i*5+j]);
    //     }

    //     printf("\n");
    // }

    // printf ("\n Rows:\n");

    // for (int i = 0; i < N*M; i++){
        
    //     printf ("%d\n", I[i]);       

    // }


    val[0] = 5;
    val[1] = 8;
    val[2] = 3;
    val[3] = 1;
    val[4] = 3;
    val[5] = 6;

    J[0] = 1;
    J[1] = 2;
    J[2] = 4;
    J[3] = 1; 
    J[4] = 4;
    J[5] = 3; 

    I[0] = 1;
    I[1] = 2;
    I[2] = 4;
    I[3] = 6;
    I[4] = 7;
 


    // x = (double*)malloc(sizeof(float)*N);
    // rhs = (double*)malloc(sizeof(float)*N);

    // for (int i = 0; i < N; i++)
    // {
    //     rhs[i] = 1.0;
    //     x[i] = 0.0;
    // }

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

    csrilu02Info_t info_coeff = 0;
    csric02Info_t info_coeff_2 = 0;
    // csrsv2Info_t   info_Lc = 0;
    // csrsv2Info_t   info_Uc = 0;
    int pBufferSize_coeff;
    int pBufferSize_Lc;
    int pBufferSize_Uc;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;

    const cusparseSolvePolicy_t policy_coeff = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lc = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Uc = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_Lc  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Uc  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;


    checkCudaErrors(cusparseStatus);

    checkCudaErrors(cudaMalloc((int **)&d_col, 6*sizeof(int)));
    checkCudaErrors(cudaMalloc((int **)&d_row, 5*sizeof(int)));
    checkCudaErrors(cudaMalloc((double **)&d_val, 6*sizeof(double)));
    // checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    // checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    // checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    // checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_col, J, 6*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, I, 5*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val,6*sizeof(double), cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice));
    

    printf ("\n Matrix:\n");

    for (int i = 0; i< 6; i++){
        printf ("%f\t", val[i]);
    }



    
    printf ("\n Columns:\n");

    for (int i = 0; i < 6; i++){
        printf ("%d\t", J[i]);
    }

 
     printf ("\n Rows:\n");

    for (int i = 0; i < 5; i++){
        printf ("%d\t", I[i]);
    }


    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    //checkCudaErrors(cusparseStatus);


    // cusparseCreateMatDescr(&descr_Lc);
    // cusparseSetMatIndexBase(descr_Lc, CUSPARSE_INDEX_BASE_ONE);
    // cusparseSetMatType(descr_Lc, CUSPARSE_MATRIX_TYPE_GENERAL);
    // cusparseSetMatFillMode(descr_Lc, CUSPARSE_FILL_MODE_LOWER);
    // cusparseSetMatDiagType(descr_Lc, CUSPARSE_DIAG_TYPE_NON_UNIT);


    // cusparseCreateMatDescr(&descr_Uc);
    // cusparseSetMatIndexBase(descr_Uc, CUSPARSE_INDEX_BASE_ONE);
    // cusparseSetMatType(descr_Uc, CUSPARSE_MATRIX_TYPE_GENERAL);
    // cusparseSetMatFillMode(descr_Uc, CUSPARSE_FILL_MODE_UPPER);
    // cusparseSetMatDiagType(descr_Uc, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csric02 and two info's for csrsv2
    checkCudaErrors(cusparseCreateCsric02Info(&info_coeff_2));
    // cusparseCreateCsrsv2Info(&info_Lc);
    // cusparseCreateCsrsv2Info(&info_Uc);
    checkCudaErrors(cusparseStatus);

    // step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
    checkCudaErrors(cusparseDcsric02_bufferSize(handle, 4, 6,
        descr_coeff, d_val, d_row, d_col, info_coeff_2, &pBufferSize_coeff));

    pBufferSize = pBufferSize_coeff; 

    checkCudaErrors(cudaMalloc((void**)&pBuffer, pBufferSize));
    //(checkCudaErrors(cusparseStatus));


    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csric02 and csrsv2 simultaneously.

    checkCudaErrors(cusparseDcsric02_analysis(handle, 4, 6,
        descr_coeff, d_val, d_row, d_col,info_coeff_2, 
        policy_coeff, pBuffer));

    cusparseStatus = cusparseXcsric02_zeroPivot(handle, info_coeff_2, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);
    // if (CUSPARSE_STATUS_ZERO_PIVOT == status){
    // printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    // }
    //checkCudaErrors(cusparseStatus);


    //step 5: M = L * L'
    // checkCudaErrors(cusparseDcsrilu02(handle, 4, 6,
    //     descr_coeff, d_val, d_row, d_col,info_coeff, 
    //     policy_coeff, pBuffer));
    // checkCudaErrors(cusparseStatus);

    // cusparseStatus = cusparseXcsrilu02_zeroPivot(handle, info_coeff, &numerical_zero);
    // if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
    //     printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    // }

    checkCudaErrors(cudaMemcpy( J, d_col,6*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( I, d_row,5*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy( val, d_val, 6*sizeof(double), cudaMemcpyDeviceToHost));

    printf ("\n Matrix:\n");

    for (int i = 0; i< 6; i++){
        printf ("%f\t", val[i]);
    }



    
    printf ("\n Columns:\n");

    for (int i = 0; i < 6; i++){
        printf ("%d\t", J[i]);
    }

 
     printf ("\n Rows:\n");

    for (int i = 0; i < 5; i++){
        printf ("%d\t", I[i]);
    }
 
}




    // alpha = 1.0;
    // alpham1 = -1.0;
    // beta = 0.0;
    // r0 = 0.;

    // printf ("\n Matrix:\n");

    // for (int i = 0; i < N*M; i++){
    //     for (int j = 0; j< 5; j++){
    //         printf ("%f\t", val[i*5+j]);
    //     }

    //     printf("\n");
    // }

    
    // printf ("\n Columns:\n");

    // for (int i = 0; i < N*M; i++){
    //     for (int j = 0; j< 5; j++){
    //         printf ("%d\t", J[i*5+j]);
    //     }

    //     printf("\n");
    // }

    // printf ("\n Rows:\n");

    // for (int i = 0; i < N*M; i++){
        
    //     printf ("%d\n", I[i]);       

    //}
    // cusparseSbsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

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

    // cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

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
//}
