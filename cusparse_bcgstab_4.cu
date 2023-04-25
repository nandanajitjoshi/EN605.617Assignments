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

void* getBSRDims(cusparseHandle_t handle, int* rowPtr, int* colPtr, double* val, 
        int* rowBSR, int*nnzb, 
        int rows,  int dimBlock
        ){


    cusparseMatDescr_t descr_coeff;
    cusparseMatDescr_t descr_coeff_2;
    
    int bufferSize = 0; 
    static void *pBuffer; 

    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff_2));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff_2, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff_2, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseDcsr2gebsr_bufferSize(handle, dir_coeff, rows, rows,  descr_coeff, 
    val, rowPtr, colPtr, dimBlock, dimBlock, &bufferSize));

    checkCudaErrors(cudaMalloc((void**)&pBuffer, bufferSize));

    checkCudaErrors(cusparseXcsr2gebsrNnz(handle, dir_coeff, rows, rows, descr_coeff, rowPtr, colPtr,
    descr_coeff_2, rowBSR, dimBlock, dimBlock, nnzb, pBuffer));

    return pBuffer; 

}

void BCGSolve(double* X, double* RHS, int* rowBSR, int* colBSR, double*valBSR, 
        cusparseHandle_t handle,  cusparseMatDescr_t descr_coeff, 
        cublasHandle_t handleBlas, 
        int mb, int nb, int nnzb, int vecSize, int maxit  ){

        double *R, *V, *T, *rw, *p;
        int dimBlock = 2; 

        double alpha = 1 ;
        double beta ;
        double alph;
        double bet;
        double omega = 1;
        double residual1 = 0;
        double residual2 = 0;
        double temp = 1;
        double rhop = 1; 
        double rho = 1;

        checkCudaErrors(cudaMalloc((double **)&R, (vecSize)*sizeof(double)));
        checkCudaErrors(cudaMalloc((double **)&p, (vecSize)*sizeof(double)));
        checkCudaErrors(cudaMalloc((double **)&rw, (vecSize)*sizeof(double)));
        checkCudaErrors(cudaMalloc((double **)&V, (vecSize)*sizeof(double)));
        checkCudaErrors(cudaMalloc((double **)&T, (vecSize)*sizeof(double)));

        checkCudaErrors(cudaMemset(V, 0, vecSize)); 


        const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
        const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

        double* Y = (double*)malloc(vecSize*sizeof(double)); 


        /*r = b - r [Ax]*/ 
        alph = -1;
        bet = 0;
   
        // printf ("Entered here");

        checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
                                      descr_coeff,valBSR, rowBSR, colBSR, dimBlock,X, 
                                      &bet, R));


        alph = 1;  
        checkCudaErrors( cublasDaxpy( handleBlas, vecSize, 
                                     &alph,RHS, 1,R, 1));

        checkCudaErrors (cudaMemcpy(Y, R,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        // printf ("\n R matrix:\n");
        // for (int i = 0; i < vecSize; i++){
            
        //     printf ("%f\n", Y[i]);       

        // }

    




    //2: Set p=r and \tilde{r}=r
    checkCudaErrors(cublasDcopy(handleBlas,(vecSize), R, 1, p, 1));
    checkCudaErrors(cublasDcopy(handleBlas, (vecSize), R, 1, rw,1));
    checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), R, 1, &residual1));  //Changed from row*cols to vecSize
    printf (" Residual %f \n", residual1);


    //3: repeat until convergence (based on max. it. and relative residual)
    for (int i=0; i<maxit; i++){
        rhop = rho; 
        //Step 5.1 : Dot product (rw,r)
        checkCudaErrors(cublasDdot ( handleBlas, vecSize, rw, 1, R, 1, &rho));
        // printf ("\n Dot Product %f \n", rho);
        printf("Iteration %d \n",i);

        if (i > 0){
            //5.2: \beta = (\rho_{i} / \rho_{i-1}) ( \alpha / \omega )
            beta= (rho/rhop)*(alpha/omega);
            printf ("\n Beta %f \n", beta);

            //5.3: p = r + \beta (p - \omega v)

            //-omega*v
            omega = -omega; 
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                            &omega,V, 1,p, 1));

            //Reset omega
            omega = -omega;


            //beta * (p - omega*v)
            checkCudaErrors(cublasDscal(handleBlas, vecSize, 
                                &beta,p, 1)); 

            // r + beta*(p-omega*v)
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                        &alph,R, 1,p, 1)); 


            checkCudaErrors (cudaMemcpy(Y, p,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

            // printf ("\n Updated P  matrix:\n");
            // for (int i = 0; i < vecSize; i++){
            
            //     printf ("%f\n", Y[i]);       

            // }
        }
            
            //Step 5.4 : v = A*p
            checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
                descr_coeff,valBSR, rowBSR, colBSR, dimBlock,p, 
                &bet, V));


            checkCudaErrors (cudaMemcpy(Y, V,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

            // printf ("\n V matrix:\n");
            // for (int i = 0; i < vecSize; i++){
                
            //     printf ("%f\n", Y[i]);       

            // }

            //Step 5.5 : alpha = rho_i/(r_tilde * v_i)

            // alpha = (r_tilde * v_i)
            checkCudaErrors(cublasDdot ( handleBlas, vecSize, rw, 1, V, 1, &alpha));

            // printf ("\n Alpha %f \n", alpha);

            //alpha = rho/alpha
            alpha = rho/alpha; 

            // printf ("\n Alpha 2 %f \n", alpha);

            /*Step 5.6/ s = r - \alpha * v */
            alpha = -alpha; 
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                        &alpha,V, 1,R, 1));

            checkCudaErrors (cudaMemcpy(Y, R,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

            // printf ("\n Updated R matrix:\n");
            // for (int i = 0; i < vecSize; i++){
                
            //     printf ("%f\n", Y[i]);       

            // }

            //Reset alpha
            alpha = -alpha; 

            /*Step 5.5 Y = Y + p*alpha*/
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                &alpha,p, 1,X, 1));

            /*Step 5.7 : Check the residual of s (r)*/
            checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), R, 1, &residual2));

            if (residual2/residual1 < 1E-3){
                break;
            }

        
        /*Step 5.9 T = A*s(r)*/
        checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
            descr_coeff,valBSR, rowBSR, colBSR, dimBlock,R, 
            &bet, T));


        checkCudaErrors (cudaMemcpy(Y, T,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        // printf ("\n Updated T matrix:\n");
        // for (int i = 0; i < vecSize; i++){
            
        //     printf ("%f\n", Y[i]);       

        // }

        /*Step 5.10 omega = (T.T)/(T.R)*/
        checkCudaErrors(cublasDdot ( handleBlas, vecSize, T, 1, T, 1, &temp));  //Changed from row*cols to vecSize
        checkCudaErrors(cublasDdot ( handleBlas, vecSize, R, 1, T, 1, &omega));  //Changed from row*cols to vecSize

        // printf ("\n Omega %f \n", omega);
        omega = omega/temp; 
        // printf ("\n Omega 2%f \n", omega);

        //Step 5.11 *x = h + omega *s*/  

        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                    &omega,R, 1,X, 1));


        /*Step 5.13 r = s - omega * t*/

        omega = -omega; 
        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                    &omega,T, 1,R, 1));
        //Reset omega
        omega = -omega; 

        checkCudaErrors (cudaMemcpy(Y, R,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        // printf ("\n Updated Final Residual matrix:\n");
        // for (int i = 0; i < vecSize; i++){
            
        //     printf ("%f\n", Y[i]);       

        // }

        /*Step 5.12 Check residual of R*/
        checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), R, 1, &residual2));

         printf ("\n Residual %f \n", residual2/residual1);

        if (residual2/residual1 < 1E-3){
            break;
        }
    }


}


void LinearSolve( int* rowPtr, int* colPtr, double* val, 
                  double* Soln, double* RHS,                   
                  int rows, int nz, int maxit ){
    //double*val_host = NULL;
    //double*val = NULL;
    double*val_BSR = NULL;
    //int* row_host, *row, *row_BSR, *col_host, *col, *col_BSR; 
    double* d_Y, *d_X; 
    int* row_BSR, *col_BSR; 
    int dimBlock = 2;


   

    int nRows = rows+1;
    int mb = (rows + dimBlock-1)/dimBlock;
    int nb = (rows + dimBlock-1)/dimBlock; 
    int base; 
    int vecSize = rows + dimBlock-1;

    checkCudaErrors(cudaMalloc((double **)&d_X, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_Y, (vecSize)*sizeof(double)));
    double* host = (double*)malloc(vecSize*sizeof(double)); 



    int bufferSize;
    void *pBuffer;
    int nnzb = 0; 
  

    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    
    cublasHandle_t handleBlas = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handleBlas);

    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

    cusparseMatDescr_t descr_coeff;
    cusparseMatDescr_t descr_coeff_2;
    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff_2));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff_2, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff_2, CUSPARSE_MATRIX_TYPE_GENERAL));


    checkCudaErrors(cudaMalloc((void**)&row_BSR, sizeof(int) *(mb+1)));



    pBuffer =  getBSRDims(handle, rowPtr, colPtr, val, row_BSR, &nnzb,
                    rows, dimBlock); 

    // printf("%d\n",nnzb);
    // printf("%d\n", nz);


    checkCudaErrors(cudaMalloc((void**)&col_BSR, sizeof(int)*(nnzb)));
    checkCudaErrors(cudaMalloc((void**)&val_BSR, sizeof(double)*(dimBlock*dimBlock)*(nnzb)));

    checkCudaErrors(cusparseDcsr2gebsr(handle, dir_coeff,rows, rows, descr_coeff, 
       val, rowPtr, colPtr, descr_coeff_2, val_BSR, row_BSR, col_BSR,        
        dimBlock, dimBlock, pBuffer));

    checkCudaErrors(cublasDcopy(handleBlas,rows, RHS, 1, d_X, 1));
    checkCudaErrors(cublasDcopy(handleBlas, rows, Soln, 1, d_Y,1));

    checkCudaErrors (cudaMemcpy(host, d_X,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

    // printf ("\n R matrix:\n");
    // for (int i = 0; i < vecSize; i++){
            
    //     printf ("%f\n", host[i]);       

    // }



    BCGSolve(d_Y,  d_X,  row_BSR,  col_BSR, val_BSR, 
              handle, descr_coeff_2, handleBlas, 
               mb, nb, nnzb, vecSize, maxit); 

    checkCudaErrors(cublasDcopy(handleBlas, rows, d_Y, 1, Soln,1));

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