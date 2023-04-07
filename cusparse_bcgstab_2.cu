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

          I[i] = 5 + I[i-1]; 
    }
}
int main(){
    double*val_host = NULL;
    double*val = NULL;
    double*val_BSR = NULL;
    int* row_host, *row, *row_BSR, *col_host, *col, *col_BSR; 
    int rows, cols; 
    int dimBlock = 2;
    rows = 5;
    cols = 5;

    int nRows = rows*cols+1;
    int nz = 5*rows*cols;
    int mb = (rows*cols + dimBlock-1)/dimBlock;
    int nb = (rows*cols + dimBlock-1)/dimBlock; 
    int base; 
    int vecSize = rows*cols + dimBlock-1;

    int bufferSize;
    void *pBuffer;
    int nnzb = 0; 
    int maxit = 8;

    int *rowHostBSR, *colHostBSR;
    double *valHostBSR = NULL; 

    double alpha = 1 ;
    double beta ;
    double alph;
    double bet;
    double omega = 1;
    double residual1 = 0;
    double residual2 = 0;
    double temp = 1;

    /*d_R is the residual, d_Y is the initial guess, and d_X is the input*/
    double * X, *d_X, *Y, *d_Y, *d_R, *d_V, *d_T;

    /*d_rw is r-tilde, d_p is p*/
    double* d_rw, *d_p;

    double rhop = 1; 
    double rho = 1;

    X = (double*)malloc(sizeof(double)*(vecSize ));
    Y = (double*)malloc(sizeof(double)*vecSize);
    row_host= (int *)malloc(sizeof(int)*(nRows+1));
    col_host = (int *)malloc(sizeof(int)*nz);
    val_host = (double *)malloc(sizeof(double) * nz);

    genTridiag (row_host, col_host, val_host, rows, cols);

    for (int i=0; i<rows*cols; i++){
        X[i] = 1.0;
    }

    for (int i = 0; i < vecSize; i++){
        Y[i] = 0.0;
    }

    // printf ("Matrx original \n");

    // for (int i = 0; i < nz; i++){
    //     printf ("%f\t", val_host[i]);
    //     if (((i+1)%5)==0){
    //         printf("\n");
    //     }   
    // }


    
    // printf ("\n Columns:\n");

    // for (int i = 0; i < nz; i++){

    //     printf ("%d\t", col_host[i]);
    //     if (((i+1)%5)==0){
    //             printf("\n");
    //     }   
    //     }

    // printf ("\n Rows:\n");

    // for (int i = 0; i < nRows; i++){
        
    //     printf ("%d\n", row_host[i]);       

    // }

    checkCudaErrors(cudaMalloc((int **)&col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((int **)&row, (nRows+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((double **)&val, nz*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_X, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_Y, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_R, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_p, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_rw, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_V, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_T, (vecSize)*sizeof(double)));

    checkCudaErrors(cudaMemset(d_V, 0, vecSize)); 
    checkCudaErrors(cudaMemset(d_Y, 0, vecSize)); 
    //checkCudaErrors(cudaMemset(d_R, 0, vecSize)); 
    //checkCudaErrors(cudaMemset(d_rw, 0, vecSize)); 
    //checkCudaErrors(cudaMemset(d_p, 0, vecSize)); 

   
    checkCudaErrors (cudaMemcpy(col, col_host, nz*sizeof(int), cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(val, val_host, nz*sizeof(double), cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(row, row_host, (nRows+1)*sizeof(int), cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(d_X, X,(vecSize)*sizeof(double), cudaMemcpyHostToDevice ));

    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    cusparseMatDescr_t descr_coeff;
    cusparseMatDescr_t descr_coeff_2;

    cublasHandle_t handleBlas = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handleBlas);


    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff_2));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff_2, CUSPARSE_INDEX_BASE_ONE));
    checkCudaErrors(cusparseSetMatType(descr_coeff_2, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseDcsr2gebsr_bufferSize(handle, dir_coeff, rows*cols, rows*cols,  descr_coeff, 
        val, row, col, dimBlock, dimBlock, &bufferSize));

    checkCudaErrors(cudaMalloc((void**)&pBuffer, bufferSize));
    checkCudaErrors(cudaMalloc((void**)&row_BSR, sizeof(int) *(mb+1)));


        
       

    checkCudaErrors(cusparseXcsr2gebsrNnz(handle, dir_coeff, rows*cols, rows*cols, descr_coeff, row, col,
        descr_coeff_2, row_BSR, dimBlock, dimBlock, &nnzb, pBuffer));

    // if (NULL != nnzb){
    //     nz = nnzb;
    // }else{
        // checkCudaErrors(cudaMemcpy(&nz, row_BSR+mb, sizeof(int), cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaMemcpy(&base, row_BSR, sizeof(int), cudaMemcpyDeviceToHost));
        // nz -= base;
    // }

    printf("%d\n",nnzb);
    printf("%d\n", nz);
    checkCudaErrors(cudaMalloc((void**)&col_BSR, sizeof(int)*(nnzb)));
    checkCudaErrors(cudaMalloc((void**)&val_BSR, sizeof(double)*(dimBlock*dimBlock)*(nnzb)));

    checkCudaErrors(cusparseDcsr2gebsr(handle, dir_coeff,rows*cols, rows*cols, descr_coeff, 
       val, row, col, descr_coeff_2, val_BSR, row_BSR, col_BSR,        
        dimBlock, dimBlock, pBuffer));

    rowHostBSR= (int *)malloc(sizeof(int)*(mb+1));
    colHostBSR = (int *)malloc(sizeof(int)*nnzb);
    valHostBSR = (double *) malloc(sizeof(double) * nz );

    checkCudaErrors (cudaMemcpy(colHostBSR, col_BSR, sizeof(int)*(nnzb), cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy(valHostBSR, val_BSR, sizeof(double) * (dimBlock*dimBlock)*(nnzb), cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy(rowHostBSR, row_BSR, (mb+1)*sizeof(int), cudaMemcpyDeviceToHost ));




// checkCudaErrors(cublasStatus);

/*Compute initial residual r = -Ax0*/

alph = -1;
bet = 0;
   
printf ("Entered here");
checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
       descr_coeff_2,val_BSR, row_BSR, col_BSR, dimBlock,d_Y, 
       &bet, d_R));




 alph = 1;  
 /*r = b - r*/  
 checkCudaErrors( cublasDaxpy( handleBlas, vecSize, 
                            &alph,d_X, 1,d_R, 1));

checkCudaErrors (cudaMemcpy(Y, d_R,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

printf ("\n R matrix:\n");
for (int i = 0; i < vecSize; i++){
    
    printf ("%f\n", Y[i]);       

}



//2: Set p=r and \tilde{r}=r
checkCudaErrors(cublasDcopy(handleBlas,(vecSize), d_R, 1, d_p, 1));
checkCudaErrors(cublasDcopy(handleBlas, (vecSize), d_R, 1, d_rw,1));
checkCudaErrors(cublasDnrm2(handleBlas,(rows*cols), d_R, 1, &residual1));



printf (" Residual %f \n", residual1);
//3: repeat until convergence (based on max. it. and relative residual)
for (int i=0; i<maxit; i++){
    rhop = rho; 
    checkCudaErrors(cublasDdot ( handleBlas, vecSize, d_rw, 1, d_R, 1, &rho));

    printf ("\n Dot Product %f \n", rho);
    if (i > 0){
        //12: \beta = (\rho_{i} / \rho_{i-1}) ( \alpha / \omega )
        beta= (rho/rhop)*(alpha/omega);

        printf ("\n Beta %f \n", beta);
        //13: p = r + \beta (p - \omega v)
        omega = -omega; 
        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                           &omega,d_V, 1,d_p, 1));
        omega = -omega;



        checkCudaErrors(cublasDscal(handleBlas, vecSize, 
                            &beta,d_p, 1)); 


        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                    &alph,d_R, 1,d_p, 1)); 


        checkCudaErrors (cudaMemcpy(Y, d_p,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        printf ("\n Updated P  matrix:\n");
        for (int i = 0; i < vecSize; i++){
        
            printf ("%f\n", Y[i]);       

        }
    }
        
        //v = A*p
        checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
            descr_coeff_2,val_BSR, row_BSR, col_BSR, dimBlock,d_p, 
            &bet, d_V));

        // alpha = rho_i/(r_tilde * v_i)

        checkCudaErrors (cudaMemcpy(Y, d_V,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        printf ("\n V matrix:\n");
        for (int i = 0; i < vecSize; i++){
            
            printf ("%f\n", Y[i]);       

        }

        //checkCudaErrors(cublasDotEx ( handleBlas, vecSize, d_rw, CUDA_R_64F,
                            //1, d_V, CUDA_R_64F, 1, &alpha, CUDA_R_64F, CUDA_R_64F));

        checkCudaErrors(cublasDdot ( handleBlas, vecSize, d_rw, 1, d_V, 1, &alpha));

        printf ("\n Alpha %f \n", alpha);


        alpha = rho/alpha; 

        printf ("\n Alpha 2 %f \n", alpha);

        //18: s = r - \alpha q

        alpha = -alpha; 
        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                    &alpha,d_V, 1,d_R, 1));

        checkCudaErrors (cudaMemcpy(Y, d_R,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        printf ("\n Updated R matrix:\n");
        for (int i = 0; i < vecSize; i++){
            
            printf ("%f\n", Y[i]);       

        }

        alpha = -alpha; 

        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
            &alpha,d_p, 1,d_Y, 1));

        checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), d_R, 1, &residual2));

        if (residual2/residual1 < 1E-3){
            break;
        }



    checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
        descr_coeff_2,val_BSR, row_BSR, col_BSR, dimBlock,d_R, 
        &bet, d_T));


    checkCudaErrors (cudaMemcpy(Y, d_T,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

    printf ("\n Updated T matrix:\n");
    for (int i = 0; i < vecSize; i++){
        
        printf ("%f\n", Y[i]);       

    }
    checkCudaErrors(cublasDdot ( handleBlas, rows*cols, d_T, 1, d_T, 1, &temp));
    checkCudaErrors(cublasDdot ( handleBlas, rows*cols, d_R, 1, d_T, 1, &omega));

     printf ("\n Omega %f \n", omega);
     omega = omega/temp; 

//     /*x = h + omega *s*/

   
    printf ("\n Omega 2%f \n", omega);

    checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                &omega,d_R, 1,d_Y, 1));


    /*r = s - omega * t*/

    omega = -omega; 
    checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                &omega,d_T, 1,d_R, 1));

    omega = -omega; 

    checkCudaErrors (cudaMemcpy(Y, d_R,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

    printf ("\n Updated Final Residual matrix:\n");
    for (int i = 0; i < vecSize; i++){
        
        printf ("%f\n", Y[i]);       

    }

    /*Check for convergence*/
    checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), d_R, 1, &residual2));

    if (residual2/residual1 < 1E-3){
        break;
    }
}





}
/*Convert CSR to BSR*/
/*Initialize x, b, ro and r_i*/
/*rho_i = dotproduct (r0',ri-1)*/
/*Beta = rho_i/rho_iminusone * alphs/omegaiminusone*/
/*rho_i = r_i+beta*(p_i-1 - omega_i-1*v_iminusone)*/
/*Matrix vector multiply - A and rho_i*/
/*alpha = rho_i/(r_o',vi) - Scalar dot prodyct*/
/*h = x + alpha * p_i*/
/*check residual b/w h and x*/
/*s = r_iinusone - alpha*v_i*/
/*t = A*s - Matrix vector multiply*/
/*omega_i = dot(t,s)/dot(t,t)*/
/*x_i = h + omega_i*s*/
/*r = s-omega*t*/