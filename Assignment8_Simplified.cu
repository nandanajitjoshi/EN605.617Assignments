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
    printf ("Entered the function");

    I[0] = 1;
    int start = 0;   //Holds the index for the vector in CSR//

    for (int i = 1; i <= rows; i++)
    {
        //start = (i-1)*3;
        
        J[i-1] = i;
        val[i-1] = 1;

        I[i] = 1 + I[i-1]; 
    }
}

// /* Converts to BSR as cusparse onlly supports BSR matrix solves*/
// void convertToBsr ( const int *I, const int *J, const double*val, 
//                     int **I_BSR,int ** J_BSR, double ** val_BSR, 
//                     int dimBlock, cusparseHandle_t handle, int rows ){

//     cusparseMatDescr_t descr_coeff;
//     const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

//     checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
//     checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
//     checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));


//     int bufferSize;
//     void *pBuffer;
//     int nnzb; 
//     int mb = (rows + dimBlock-1)/dimBlock;

//     checkCudaErrors(cusparseDcsr2gebsr_bufferSize(handle, dir_coeff, rows, rows,  descr_coeff, 
//         val, I, J, dimBlock, dimBlock, &bufferSize));

//     checkCudaErrors(cudaMalloc((void**)&pBuffer, bufferSize));
//     checkCudaErrors(cudaMalloc((void**)I_BSR, sizeof(int) *(mb+1)));


            
        

//     checkCudaErrors(cusparseXcsr2gebsrNnz(handle, dir_coeff, rows, rows, descr_coeff, I, J,
//         descr_coeff, *I_BSR, dimBlock, dimBlock, &nnzb, pBuffer));

//         // if (NULL != nnzb){
//         //     nz = nnzb;
//         // }else{
//             // checkCudaErrors(cudaMemcpy(&nz, row_BSR+mb, sizeof(int), cudaMemcpyDeviceToHost));
//             // checkCudaErrors(cudaMemcpy(&base, row_BSR, sizeof(int), cudaMemcpyDeviceToHost));
//             // nz -= base;
//         // }



//         printf("%d\n",nnzb);
//         printf("%d\n", nz);
//         checkCudaErrors(cudaMalloc((void**)J_BSR, sizeof(int)*(nnzb)));
//         checkCudaErrors(cudaMalloc((void**)val_BSR, sizeof(double)*(dimBlock*dimBlock)*(nnzb)));

//         checkCudaErrors(cusparseDcsr2gebsr(handle, dir_coeff,rows, rows, descr_coeff, 
//         val, I, J, descr_coeff, *val_BSR, *I_BSR, *J_BSR,        
//             dimBlock, dimBlock, pBuffer));
// }

int main(){

    int *I, *J; 
    double *val_host ; 
    double* RHS = NULL ; 
    double* Soln = NULL; 

    int * I_dvc, *J_dvc; 
    double* val_dvc = NULL; 
    double *RHS_dvc = NULL; 
    double *Soln_Dvc = NULL; 

   

    int rows = 5; 
    int dimBlock = 1;   //CSR = BSR wth block dimension 1

    I = (int *)malloc(sizeof(int)*(rows+1));
    J = (int *)malloc(sizeof(int)*rows);
    val_host = (double *)malloc(sizeof(double) * rows);
    RHS = (double *)malloc(sizeof(double) * rows);
    Soln = (double *)malloc(sizeof(double) * rows);


  

    /*Generate the sparse coefficient matrix*/
    genTridiag (I, J, val_host, rows);

    for (int i=0; i<rows; i++){
        RHS[i] = 2*(double)i;
    }

    // for (int i = rows; i < vecSize; i++){
    //     Soln[i] = 0.0;
    // }


    printf ("\n Values:\n");
    for (int i = 0; i < rows; i++){
        printf ("%f\t", val_host[i]);
        if (((i+1)%3)==0){
            printf("\n");
        }   
    }


    
    printf ("\n Columns:\n");

    for (int i = 0; i < rows; i++){

        printf ("%d\t", J[i]);
        if (((i+1)%3)==0){
                printf("\n");
        }   
        }

    printf ("\n Rows:\n");

    for (int i = 0; i < rows +1; i++){
        
        printf ("%d\n", I[i]);       

    }



    checkCudaErrors(cudaMalloc((int **)&J_dvc, rows*sizeof(int)));
    checkCudaErrors(cudaMalloc((int **)&I_dvc, (rows+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((double **)&val_dvc, rows*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&RHS_dvc, (rows)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Soln_Dvc, (rows)*sizeof(double)));

    checkCudaErrors (cudaMemcpy(J_dvc, J,  rows*sizeof(int), cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(val_dvc, val_host, rows*sizeof(int), cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(I_dvc, I, (rows+1)*sizeof(int), cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy( RHS_dvc, RHS,(rows)*sizeof(double), cudaMemcpyHostToDevice ));



    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    cusparseMatDescr_t descr_coeff = 0;
    bsrsv2Info_t  info_coeff = 0;
    const cusparseSolvePolicy_t policy_coeff = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_COLUMN;
    int pBufferSize; 
    void* pBuffer = 0;
    double alpha = 1;
    int structural_zero, numerical_zero;

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));
    //cusparseSetMatFillMode(descr_coeff, CUSPARSE_FILL_MODE_UPPER);
    //cusparseSetMatDiagType(descr_coeff, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateBsrsv2Info(&info_coeff);



    //checkCudaErrors (cusparseStatus);

    checkCudaErrors(cusparseDbsrsv2_bufferSize(handle, dir_coeff, trans_coeff, rows, rows,
                 descr_coeff, val_dvc,I_dvc, J_dvc, dimBlock, info_coeff, &pBufferSize));

    

    // checkCudaErrors(cudaMalloc((int **)&J_dvc, rows*sizeof(int)));
    // checkCudaErrors(cudaMalloc((int **)&I_dvc, (rows+1)*sizeof(int)));
    // checkCudaErrors(cudaMalloc((double **)&val_dvc, rows*sizeof(double)));
    // checkCudaErrors(cudaMalloc((double **)&RHS_dvc, (rows)*sizeof(double)));
    // checkCudaErrors(cudaMalloc((double **)&Soln_Dvc, (rows)*sizeof(double)));
    // //checkCudaErrors(cudaMemset(Soln_Dvc, 0, rows)); 


   
    // checkCudaErrors (cudaMemcpy(J_dvc, J,  rows*sizeof(int), cudaMemcpyHostToDevice ));
    // checkCudaErrors (cudaMemcpy(val_dvc, val_host, rows*sizeof(int), cudaMemcpyHostToDevice ));
    // checkCudaErrors (cudaMemcpy(I_dvc, I, (rows+1)*sizeof(int), cudaMemcpyHostToDevice ));
    // checkCudaErrors (cudaMemcpy( RHS_dvc, RHS,(rows)*sizeof(double), cudaMemcpyHostToDevice ));
  
    // //checkCudaErrors (cusparseStatus);

    

    // checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    // checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    // checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    //  checkCudaErrors (cusparseStatus);


    // checkCudaErrors(cusparseDbsrsv2_bufferSize(handle, dir_coeff, trans_coeff, rows, rows,
    //             descr_coeff, val_dvc,I_dvc, J_dvc, dimBlock, info_coeff, &pBufferSize));

    // checkCudaErrors (cudaMemcpy( RHS, RHS_dvc,(rows)*sizeof(double), cudaMemcpyDeviceToHost));

    // printf ("\n Output:\n");

    // for (int i = 0; i < rows ; i++){
        
    //     printf ("%f\n", Soln[i]);       

    // }

    checkCudaErrors(cudaMalloc((void**)&pBuffer, pBufferSize));

    checkCudaErrors(cusparseDbsrsv2_analysis(handle, dir_coeff, trans_coeff, rows, rows,
                descr_coeff, val_dvc,I_dvc, J_dvc, dimBlock, info_coeff, policy_coeff, pBuffer));



    cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        printf("U(%d,%d) is Sstructural zero\n", structural_zero, structural_zero);
    }

    checkCudaErrors(cusparseDbsrsv2_solve(handle, dir_coeff, trans_coeff, rows, rows, &alpha,
               descr_coeff, val_dvc,I_dvc, J_dvc, dimBlock, info_coeff, RHS_dvc, Soln_Dvc,
               policy_coeff, pBuffer));


    cusparseStatus = cusparseXbsrsv2_zeroPivot(handle, info_coeff, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus){
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    checkCudaErrors (cudaMemcpy( Soln, Soln_Dvc,(rows)*sizeof(double), cudaMemcpyDeviceToHost));

    printf ("\n Output:\n");

    for (int i = 0; i < rows ; i++){
        
        printf ("%f\n", Soln[i]);       

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