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

struct vertex {
    char VType; 
    char BSide; 
    char BType;
    double dely; 
    double delx; 
    double UValue; 
    double VValue; 
    double PValue; 
}; 

#define U0 1

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
void genTridiag(double *I, double *J, double*val, int rows, double mu, 
    double rho, double dely, double delx, double delT)
{



    //int start = 0;   //Holds the index for the vector in CSR//

    for (int i = 0; i < rows; i++)
    {
        I[i] = - mu/rho * (dely/delx)/(dely*delx) *delT/2;
        J[i] = - mu/rho * (dely/delx)/(dely*delx) *delT/2;
        val [i] = 1 + 2 * mu/rho * (dely/delx)/(dely*delx) *delT/2;
    }

    I[0] = 0;
    J[0] = 0;
    J[rows - 1] = 0;
    I[rows-1] = 0;
    val[rows-1] = 1;
}

/*Generates the RHS vector for linear solve
* All values are set to 1
*/
void genRHS (double* RHS, int rows, int Nx){
    for (int i=0; i < rows; i++){
        RHS[i] = (double)((i%Nx) * (i%Nx)) ;
    }
}

void genRHS2 (double* RHS, int rows, int Nx){
    for (int i=0; i < rows; i++){
        RHS[i] = (double)((i/Nx) * (i/Nx));
    }
}


__global__
void XtoY(double*RHS, int Ny, int Nx, double*RHS_y){

    int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (thread_idx < Nx*Ny){
        int row = thread_idx % Ny;
        int col =  thread_idx / Ny; 

        RHS_y[thread_idx] = RHS[row*Nx+col];
    }
    
}

__global__
void YtoX(double*RHS, int Ny, int Nx, double*RHS_y){

    int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (thread_idx < Nx*Ny){
        int row = thread_idx / Nx;
        int col =  thread_idx % Nx; 

        RHS_y[thread_idx] = RHS[col*Ny+row];
    }
    
}

/* Prints the results of the sparse solve 
* First prints the tridiagonal coefficent matrix
* Followed by the RHS and results*/

void printOutput(double *I, double *J, double*val, double* RHS, double* Soln, int rows){

    printf ("The coefficient matrix is : \n"); 
    printf ("\n");
     for (int i = 0; i < rows; i++){
        for (int j=0; j< rows; j++){
            if (j== (i-1)){

                printf ("%3.2f ", I[i]);

            } else if (j==i){
                printf ("%3.2f ", val[i]);
            } else if (j== (i+1)){
                printf ("%3.2f ", J[i]);
            } else {
                printf("0 "); 

            }


            }
            //valIndex ++;

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

float TriDiagSolve(double* low , double* diag , double* high, double* RHS, int rows){


    //int rows = 5; 
    int dimBlock = 1;   //CSR = BSR wth block dimension 1

    /*Allocate memory for host-side arrays*/


    /*Initialize the variables to be used in the linear solve*/
    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    // cusparseMatDescr_t descr_coeff = 0;
    // bsrsv2Info_t  info_coeff = 0;
    // const cusparseSolvePolicy_t policy_coeff = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    // const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW; //Doesn't matter for blockdIM = 1
    size_t pBufferSize; 
    void* pBuffer = 0;
    //double alpha = 1;
    int structural_zero, numerical_zero;    //To check for singularities in the coefficient matrix

    /*1 - Create descriptor for coeff matrix*/
    // checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    // checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ONE));
    // checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));
    // cusparseSetMatDiagType(descr_coeff, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // /*2 - Create info for linear solve*/
    // cusparseCreateBsrsv2Info(&info_coeff);


    /*Create timer variables*/
    cudaEvent_t startTime, stopTime; 
    float time; 
    cudaEventCreate (&startTime);
    cudaEventCreate (&stopTime);
    cudaEventRecord (startTime, 0);


    /*4 - Allocate buffer space for linear solve*/
    checkCudaErrors(cusparseDgtsv2_bufferSizeExt(handle, rows, 1, 
            low,diag, high, RHS ,rows , &pBufferSize));   //Why rows + 10?

    printf ("%zu", pBufferSize);
    checkCudaErrors(cudaMalloc((void**)&pBuffer, sizeof(int)*pBufferSize));

    /*5 - Analyze coefficient matrix and report any singularities */
    checkCudaErrors(cusparseDgtsv2(handle, rows, 1, 
            low,diag, high, RHS ,rows , pBuffer));

  

    /*Record end time*/
    cudaEventRecord (stopTime, 0);
    cudaEventSynchronize (stopTime);
    cudaEventElapsedTime (&time, startTime, stopTime);
    cudaEventDestroy (startTime);
    cudaEventDestroy (stopTime); 

   



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

float callDiff (int timeStep, int Nx, int Ny){

    int rows = Nx * Ny; 

    double* I = (double *)malloc(sizeof(double)*(rows));
    double *J = (double *)malloc(sizeof(double)*rows);
    double *val_host = (double *)malloc(sizeof(double) * rows );
    double *RHS = (double *)malloc(sizeof(double) * rows);
    double *Soln = (double *)malloc(sizeof(double) * rows);
  
    double mu = 1E-5;
    double rho = 1.2;
    double delx = 0.001;
    double dely = 0.001; 
    /*Generate the sparse coefficient matrix*/
    //genTridiag (I, J, val_host, rows, mu, rho, dely, delx, delT);

    /*Generate RHS vector*/
    genRHS( RHS, rows, Nx); 

    double * I_dvc, *J_dvc; 
    double* val_dvc = NULL; 
    double *RHS_dvc = NULL; 
    double *Soln_Dvc = NULL; 
    double* Sol_y = NULL; 

    float Time; 

    int blockWidth = 128;

    /*Allocate memory on the device for the arrays*/
    checkCudaErrors(cudaMalloc((double **)&J_dvc, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&I_dvc, (rows)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&val_dvc, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&RHS_dvc, (rows)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Soln_Dvc, (rows)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Sol_y, (rows)*sizeof(double)));



    /* 3 - Transfer data from host to device*/
    checkCudaErrors (cudaMemcpy(J_dvc, J,  rows *sizeof(double), 
            cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(val_dvc, val_host, rows *sizeof(double), 
            cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy(I_dvc, I, (rows)*sizeof(double),
             cudaMemcpyHostToDevice ));
    checkCudaErrors (cudaMemcpy( RHS_dvc, RHS,(rows)*sizeof(double), 
            cudaMemcpyHostToDevice ));




    Time = TriDiagSolve(I_dvc , val_dvc, J_dvc, RHS_dvc, rows);

    int nBlocks = rows/blockWidth + 1; 

    XtoY<<<nBlocks, blockWidth>>> ( RHS_dvc, Ny, Nx, Sol_y);
     
     /* 7 - Transfer result data back to host*/
    checkCudaErrors (cudaMemcpy( Soln, RHS_dvc,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    //Print output
    printOutput (I, J, val_host, RHS,Soln, rows);

    //double* Sol_y = (double *)malloc(sizeof(double) * rows);
    
    checkCudaErrors (cudaMemcpy( Soln, Sol_y,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\n The transposed vector is:\n");

    for (int i = 0; i < rows ; i++){
        
        printf ("%f\n", Soln[i]);       

    }

    TriDiagSolve(I_dvc , val_dvc, J_dvc, Sol_y, rows);

    checkCudaErrors (cudaMemcpy( Soln, Sol_y,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\n The solved vector is:\n");

    for (int i = 0; i < rows ; i++){
        
        printf ("%f\n", Soln[i]);       

    }

    YtoX<<<nBlocks, blockWidth>>> ( Sol_y, Ny, Nx, RHS_dvc);

    checkCudaErrors (cudaMemcpy( Soln, RHS_dvc,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\n The re-transposed vector is:\n");

    for (int i = 0; i < rows ; i++){
        
        printf ("%f\n", Soln[i]);       

    }


    //Free CUDA memory
    cudaFree (I_dvc);
    cudaFree (J_dvc);
    cudaFree (val_dvc);
    cudaFree (RHS_dvc);
    cudaFree (Soln_Dvc);  


    //XtoY( Soln, rows, Ny, Nx, Sol_y); 



    return Time; 
}

/*Ver 2 - Added 2*/
__global__
void genXCoeffs(double *low, double *high, double*diag, const vertex* Domain, 
     int rows, double mu, double rho, double delT){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx]; 
    double dely = thisV.dely; 
    double delx = thisV.delx; 

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/
                low [thread_idx] =  - (mu/rho * (dely/delx)/(dely*delx))*(delT/2);
                high [thread_idx] =  - mu/rho * (dely/delx)/(dely*delx) * (delT/2) ;
                diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2)  ;

                break;

            case '1':
                switch (thisV.BSide){
                    case 'L':
                    /*Left-Side Boundary*/
                        if ((thisV.BType == '0') || (thisV.BType == '2')) {
                            /*Dirichlet BC*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            low [thread_idx] =  0;
                            high [thread_idx] =  - 1*mu/rho * (dely/delx)/(dely*delx)*(delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (dely/delx)/(dely*delx)* (delT/2);

                        }
                    break;

                    case 'R':
                        /*Right-Side Boundary*/
                        if ((thisV.BType == '0')|| (thisV.BType == '2')){
                            /*Dirichlet BC*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            high [thread_idx] =  0;
                            low [thread_idx] =  - 1*mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);

                        }

                    break; 
                    default:
                    /*Top and bottom boundaries treated as interior zones*/
                        low [thread_idx] =  - mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                        high [thread_idx] =  - mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                        diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2); 

                }

            break;
            case '2':

                if (thisV.BType == '0'){

                    /*Dirichlet condition*/
                     low [thread_idx] =  0;
                     high [thread_idx] =  0;
                     diag [thread_idx] =  1;

                } else{
                    if (thisV.BSide == 'W' || thisV.BSide == 'Z' ){

                    /*Left Side points*/
                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                low [thread_idx] =  0;
                                high [thread_idx] =  - 2*mu/rho * (dely/delx)/(dely*delx) * (delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);

                        }
                    } else if ( thisV.BSide == 'X' || thisV.BSide == 'Y') {
                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                high [thread_idx] =  0;
                                low [thread_idx] =  - 2*mu/rho * (dely/delx)/(dely*delx) *(delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (dely/delx)/(dely*delx) * (delT/2);

                        }
                    }  
                }

            break;

        }

    }
}

/*Ver 2 -Added division by 2*/
__global__
void genYCoeffs(double *low, double *high, double*diag, const vertex* Domain, 
     int rows, double mu, double rho, double delT){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx]; 
    double dely = thisV.dely; 
    double delx = thisV.delx; 

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/
                low [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                high [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx)  * (delT/2);

                break;

            case '1':
                switch (thisV.BSide){
                    case 'B':
                    /*Bottom-Side Boundary*/
                        if ((thisV.BType == '0') || (thisV.BType == '2')) {
                            /*Dirichlet BC*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            low [thread_idx] =  0;
                            high [thread_idx] =  - 1*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }
                    break;

                    case 'T':
                        /*Top-Side Boundary*/
                        if ((thisV.BType == '0')|| (thisV.BType == '2')){
                            /*Dirichlet BC*/
                            low [thread_idx] =  0;
                            high [thread_idx] =  0;
                            diag [thread_idx] =  1 ;

                        } else if (thisV.BType == '1') {
                            /*Symmetry BC */
                            high [thread_idx] =  0;
                            low [thread_idx] =  - 1*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                            diag [thread_idx] =  1 + 1 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }

                    break; 
                    default:
                    /*Top and bottom boundaries treated as interior zones*/
                        low [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                        high [thread_idx] =  - mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                        diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx) * (delT/2); 

                }

            break;
            case '2':

                if (thisV.BType == '0'){

                    /*Dirichlet condition*/
                     low [thread_idx] =  0;
                     high [thread_idx] =  0;
                     diag [thread_idx] =  1;

                } else{
                    if (thisV.BSide == 'X' || thisV.BSide == 'Z' ){

                    /*Bottom Side points*/
                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                low [thread_idx] =  0;
                                high [thread_idx] =  - 2*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }
                    } else if ( thisV.BSide == 'D' || thisV.BSide == 'W') {

                        /*Top Side points*/

                        if (thisV.BType == '1') {

                                /*Symmetry BC*/
                                high [thread_idx] =  0;
                                low [thread_idx] =  - 2*mu/rho * (delx/dely)/(dely*delx) * (delT/2);
                                diag [thread_idx] =  1 + 2 * mu/rho * (delx/dely)/(dely*delx) * (delT/2);

                        }
                    }  
                }

            break;

        }

    }
}

__global__
void calcH(const double *U, const double *V, const double*vec, double* H, 
    const vertex* Domain, int rows, int Nx){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx]; 
    double dely = thisV.dely; 
    double delx = thisV.delx; 

    double right;
    double left;
    double top;
    double btm;

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/

                right = ((U[thread_idx]+ U[thread_idx+1])/2)*((vec[thread_idx]+ vec[thread_idx+1])/2);
                left = ((U[thread_idx]+ U[thread_idx-1])/2)*((vec[thread_idx]+ vec[thread_idx-1])/2);  
                top = ((V[thread_idx]+ V[thread_idx-Nx])/2)*((vec[thread_idx]+ vec[thread_idx-Nx])/2);
                btm = ((V[thread_idx]+ V[thread_idx+Nx])/2)*((vec[thread_idx]+ vec[thread_idx+Nx])/2);  
                H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/(dely*delx);
                break;

            case '1':

                if (thisV.BType == '0' || thisV.BType == '2'){
                    /*Dirichlet condition*/

                    H[thread_idx] = 0;

                } else {
                    switch (thisV.BSide){
                        case 'B':
                            /*Bottom-Side Boundary*/
                            right = ((U[thread_idx]+ U[thread_idx+1])/2)*((vec[thread_idx]+ vec[thread_idx+1])/2);
                            left = ((U[thread_idx]+ U[thread_idx-1])/2)*((vec[thread_idx]+ vec[thread_idx-1])/2);  
                            H[thread_idx] = ((right - left)*dely)/(dely*delx);
                        break;

                        case 'T':
                            /*Top-Side Boundary*/
                            right = ((U[thread_idx]+ U[thread_idx+1])/2)*((vec[thread_idx]+ vec[thread_idx+1])/2);
                            left = ((U[thread_idx]+ U[thread_idx-1])/2)*((vec[thread_idx]+ vec[thread_idx-1])/2);  
                            H[thread_idx] = ((right - left)*dely)/(dely*delx);
                            break;

                        case 'R':
                        /*Right-Side Boundary*/
                            top = ((V[thread_idx]+ V[thread_idx-Nx])/2)*((vec[thread_idx]+ vec[thread_idx-Nx])/2);
                            btm = ((V[thread_idx]+ V[thread_idx+Nx])/2)*((vec[thread_idx]+ vec[thread_idx+Nx])/2);  
                            H[thread_idx] = ( (top - btm )*delx)/(dely*delx);
                        break;

                        case 'L':
                        /*Left-Side Boundary*/
                            top = ((V[thread_idx]+ V[thread_idx-Nx])/2)*((vec[thread_idx]+ vec[thread_idx-Nx])/2);
                            btm = ((V[thread_idx]+ V[thread_idx+Nx])/2)*((vec[thread_idx]+ vec[thread_idx+Nx])/2);  
                            H[thread_idx] = ( (top - btm )*delx)/(dely*delx);
                        break;

                    }

                break;
            }
            case '2':

                if (thisV.BType == '0' || thisV.BType == '2'){

                    /*Dirichlet condition*/
                    H[thread_idx] = 0;

                } else{
                     /*Symmetry condition*/
                    H[thread_idx] = 0;
                   
                }
            break; 
        }
    }
}


/*Has to be multiplied by dT/2 and added by u/v
1 + dT/2*D*/
__global__
void calcD(const double *vec, double*H,  const vertex* Domain, int rows, int Nx, 
    char varId, double delT, double mu, double rho){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertex thisV = Domain[thread_idx]; 
    double dely = thisV.dely; 
    double delx = thisV.delx; 
    double thisVar; 


    double right;
    double left;
    double top;
    double btm;

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior vertices*/

                right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely; 
                H[thread_idx] = (((right - left)*dely + (top - btm )*delx)/(dely*delx))*delT/2 * mu/rho + vec[thread_idx];
                break;

            case '1':

                if (thisV.BType == '0' || thisV.BType == '2'){
                    /*Dirichlet condition*/
                    switch (varId){
                        case 'U':
                        H[thread_idx] = thisV.UValue;
                        break; 
                        case 'V':
                        H[thread_idx] = thisV.VValue;
                        break; 
                    }

                } else {
                    switch (thisV.BSide){
                        case 'B':
                            /*Bottom-Side Boundary*/

                            right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                            left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                            top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            btm = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/(dely*delx) * delT/2 * mu/rho + vec[thread_idx];
                        break;

                        case 'T':
                            /*Top-Side Boundary*/
                            right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                            left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                            top = (vec[thread_idx]- vec[thread_idx + Nx])/dely;
                            btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely;
                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/(dely*delx) * delT/2 *mu/rho +vec[thread_idx];

                            break;

                        case 'R':
                        /*Right-Side Boundary*/
                            right = (vec[thread_idx]- vec[thread_idx-1])/delx;
                            left = (vec[thread_idx]- vec[thread_idx-1])/delx;  
                            top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely; 
                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/(dely*delx) *delT/2 *mu/rho+vec[thread_idx];
                        break;

                        case 'L':
                        /*Left-Side Boundary*/
                            right = (vec[thread_idx+1]- vec[thread_idx])/delx;
                            left = (vec[thread_idx+1]- vec[thread_idx])/delx;  
                            top = (vec[thread_idx-Nx]- vec[thread_idx])/dely;
                            btm = (vec[thread_idx]- vec[thread_idx + Nx])/dely; 
                            H[thread_idx] = ((right - left)*dely + (top - btm )*delx)/(dely*delx) * delT/2 *mu/rho +vec[thread_idx] ;

                        break;

                    }

                break;
            }
            case '2':

                if (thisV.BType == '0' || thisV.BType == '2'){

                    /*Dirichlet condition*/
                    switch (varId){
                        case 'U':
                        H[thread_idx] = thisV.UValue;
                        break; 
                        case 'V':
                        H[thread_idx] = thisV.VValue;
                        break; 
                    }

                } else{
                     /*Symmetry condition*/
                    H[thread_idx] = 0;
                   
                }
            break; 
        }
    }
}

__global__
void Mesh (int Lx, int Ly, int Nx,int Ny, vertex* Domain){

    int rows = Ny*Nx; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < rows){
        if (i == 0){
            /* Top left corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'W';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if (i== (Ny-1)){
            /*Top Right Corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'D';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if (i==(Ny*(Nx-1))){
            /*Bottom Left Corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'Z';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if (i== (rows -1)){
            /*Bottom Right Corner*/
            Domain[i]. VType = '2';
            Domain[i].BSide = 'X';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].BType = '0';
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].PValue = 0;

        } else if  (i < Ny){
            /*Top Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'T';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/Nx;
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '1';
            Domain[i].PValue = 0;

        } else if (i%Ny == 0){
            /*Left Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'L';
            Domain[i]. dely = (double)Ly/Ny;
            Domain[i].delx = (double) Lx/(2*Nx);
            Domain[i].UValue = U0;
            Domain[i].VValue = 0;
            Domain[i].BType = '0';
            Domain[i].PValue = 0;

        } else if (i%Ny == (Ny-1)){
            /*Right Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'R';
            Domain[i]. dely = (double)Ly/Ny;
            Domain[i].delx = (double)Lx/(2*Nx);
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '0';
            Domain[i].PValue = 0;

        } else if (i >= (Ny*(Nx-1)) ){
            /*Bottom Edge*/
            Domain[i]. VType = '1';
            Domain[i].BSide = 'B';
            Domain[i]. dely = (double)Ly/(2*Ny);
            Domain[i].delx = (double)Lx/Nx;
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '1';
            Domain[i].PValue = 0;

        } else {
            /*Internal Points*/
            Domain[i]. VType = '0';
            Domain[i].BSide = '0';
            Domain[i].dely = (double)Ly/Ny;
            Domain[i].delx = (double)Lx/Nx;
            Domain[i].UValue = 0;
            Domain[i].VValue = 0;
            Domain[i].BType = '0';
            Domain[i].PValue = 0;
        } 
    }


      
}

void updateRHS(cublasHandle_t handleBlas, double* H_u_n_1, double* H_v_n_1,
     double* H_u_n, double* H_v_n, double* D_u_n, double* D_v_n, 
     double* RHS_u_n, double* RHS_v_n, double delT, double mu, double rho, 
     int rows ){

    double alpha = -1.5 *delT; 

    /*Add vectors - Y = X + Y*/
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,H_u_n_1, 1,RHS_u_n, 1));

    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                        &alpha,H_v_n_1, 1,RHS_v_n, 1));

    alpha = 0.5 *delT; 

    /*Add vectors - Y = X + Y*/
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,H_u_n, 1,RHS_u_n, 1));

    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                        &alpha,H_v_n, 1,RHS_v_n, 1));


    alpha = 1; 

    /*Add vectors - Y = X + Y*/
    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                           &alpha,D_u_n, 1,RHS_u_n, 1));

    checkCudaErrors( cublasDaxpy( handleBlas, rows, 
                        &alpha,D_v_n, 1,RHS_v_n, 1));  


    
}

__global__
void getPCoeffs(double* P_rowPtr, double* P_colPtr, double* P_val,
                 const vertex* Domain, int rows, int Nx, int Ny,
                 double rho, double delT){


    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int start = thread_idx*5; 
    vertex thisV = Domain[thread_idx]; 
    double delx = thisV.delx;
    double dely = thisV.dely; 
    double constTerm  = 1/(rho*(delx*dely))*delT;


    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
                /*Interior vertices*/
                P_colPtr[start] = thread_idx - Nx; 
                P_colPtr[start+1] = thread_idx - 1; 
                P_colPtr[start+2] = thread_idx ; 
                P_colPtr[start+3] = thread_idx+1 ; 
                P_colPtr[start+4] = thread_idx+Nx ; 


                P_val[start] = 1; 
                P_val[start+1] = 1; 
                P_val[start+2] = -4 ; 
                P_val[start+3] = 1 ; 
                P_val[start+4] = 1 ; 
                break;

            case '1':
                switch (thisV.BSide){
                    case 'B' :
                        P_colPtr[start] = thread_idx - (Ny-1)*Nx ; 
                        P_colPtr[start+1] = thread_idx - Nx; 
                        P_colPtr[start+2] = thread_idx - 1; 
                        P_colPtr[start+3] = thread_idx ; 
                        P_colPtr[start+4] = thread_idx+1 ; 
                       

                        P_val[start] = 0 ; 
                        P_val[start+1] = 2; 
                        P_val[start+2] = 1; 
                        P_val[start+3] = -4 ; 
                        P_val[start+4] = 1 ; 
                       
                    break;  

                    case 'T' :
                        P_colPtr[start] = thread_idx - 1; 
                        P_colPtr[start+1] = thread_idx; 
                        P_colPtr[start+2] = thread_idx + 1; 
                        P_colPtr[start+3] = thread_idx + Nx; 
                        P_colPtr[start+4] = thread_idx+Nx*(Ny-1) ; 
                       

                        P_val[start] = 1 ; 
                        P_val[start+1] = -4; 
                        P_val[start+2] = 1; 
                        P_val[start+3] = 2 ; 
                        P_val[start+4] = 0 ; 
                       
                    break;

                    case 'L' :
                        P_colPtr[start] = thread_idx - Nx; 
                        P_colPtr[start+1] = thread_idx; 
                        P_colPtr[start+2] = thread_idx + 1; 
                        P_colPtr[start+3] = thread_idx + Nx-1; 
                        P_colPtr[start+4] = thread_idx+Nx ; 
                       

                        P_val[start] = 1 ; 
                        P_val[start+1] = -4; 
                        P_val[start+2] = 2; 
                        P_val[start+3] = 0 ; 
                        P_val[start+4] = 1 ; 
                       
                    break;

                    case 'R' :
                        P_colPtr[start] = thread_idx - Nx; 
                        P_colPtr[start+1] = thread_idx-Nx+1; 
                        P_colPtr[start+2] = thread_idx - 1; 
                        P_colPtr[start+3] = thread_idx ; 
                        P_colPtr[start+4] = thread_idx+Nx; 
                       

                        P_val[start] = 1 ; 
                        P_val[start+1] = 0; 
                        P_val[start+2] = 2; 
                        P_val[start+3] = -4 ; 
                        P_val[start+4] = 1 ; 
                       
                    break;

                }

            case '2':
                switch (thisV.BSide){       
                    case 'W':
                    /*Top left*/

                    P_colPtr[start] = thread_idx; 
                    P_colPtr[start+1] = thread_idx+1; 
                    P_colPtr[start+2] = thread_idx+Nx-1; 
                    P_colPtr[start+3] = thread_idx + Nx ; 
                    P_colPtr[start+4] = thread_idx+Nx*(Ny-1); 


                    P_val[start] = -4 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = 0; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = 0 ; 
                    break;

                    case 'D':
                    /*Top right*/
                    P_colPtr[start] = thread_idx-Nx+1; 
                    P_colPtr[start+1] = thread_idx-1; 
                    P_colPtr[start+2] = thread_idx; 
                    P_colPtr[start+3] = thread_idx + Nx ; 
                    P_colPtr[start+4] = thread_idx+Nx*(Ny-1); 


                    P_val[start] = 0 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = -4; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = 0 ; 
                    break; 

                    case 'Z':
                    /*Bottom left*/

                    P_colPtr[start] = thread_idx-Nx*(Ny-1); 
                    P_colPtr[start+1] = thread_idx-Nx; 
                    P_colPtr[start+2] = thread_idx; 
                    P_colPtr[start+3] = thread_idx + 1 ; 
                    P_colPtr[start+4] = thread_idx+Nx-1; 


                    P_val[start] = 0 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = -4; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = 0 ; 
                    break;
                    case 'X':
                    /*Bottom Right*/

                    P_colPtr[start] = thread_idx-(Ny-1)*Nx; 
                    P_colPtr[start+1] = thread_idx-Nx; 
                    P_colPtr[start+2] = thread_idx-Nx+1; 
                    P_colPtr[start+3] = thread_idx -1 ; 
                    P_colPtr[start+4] = thread_idx; 


                    P_val[start] = 0 ; 
                    P_val[start+1] = 2; 
                    P_val[start+2] = 0; 
                    P_val[start+3] = 2 ; 
                    P_val[start+4] = -4 ; 
                    break; 
                }

        }

        P_rowPtr[thread_idx + 1] = 5*thread_idx + 4;
    }

}

__global__
void update_PRHS(double* P_RHS, double* U, double* V,
                 const vertex* Domain, int rows, int Nx, int Ny,
                 double rho){



    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    vertex thisV = Domain[thread_idx]; 
    double delx = thisV.delx;
    double dely = thisV.dely; 

    double rightV, leftV, topV, btmV; 


    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
                /*Interior vertices*/
            rightV = U[thread_idx+1];
            leftV = U[thread_idx-1];
            topV = V[thread_idx-Nx]; //Note that V vertex has been rotated in row-major direction
            btmV = V[thread_idx+Nx];

            P_RHS[thread_idx] = (rightV - leftV)/(2*delx) + (topV - btmV)/(2*dely);
            break;

            case '1':
                switch (thisV.BSide){
                    case 'B' :
                        rightV = U[thread_idx+1];
                        leftV = U[thread_idx-1];

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Dirichlet Condition*/
                        topV = V[thread_idx-Nx];                       
                        btmV =  2*thisV.VValue - topV;   
                    } else {
                        topV = 0;                       
                        btmV =  0;  
                    }

                    P_RHS[thread_idx] = (rightV - leftV)/(2*delx) + (topV - btmV)/(4*dely);                       
                    break;  

                    case 'T' :
                        /*Top Boundary*/
                        rightV = U[thread_idx+1];
                        leftV = U[thread_idx-1];

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Dirichlet Condition*/
                        btmV = V[thread_idx+Nx];                       
                        topV =  2*thisV.VValue - btmV; 
                        } else {
                        topV = 0;                       
                        btmV =  0;  
                    }

                    P_RHS[thread_idx] = (rightV - leftV)/(2*delx) + (topV - btmV)/(4*dely);                       
                    break;  

                    case 'L' :

                        topV = V[thread_idx-Nx];
                        btmV = V[thread_idx + Nx]; 

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Dirichlet Condition*/
                        rightV = U[thread_idx+1];                       
                        leftV =  2*thisV.UValue - rightV; 
                        } else {
                        rightV = 0;                       
                        leftV =  0;  
                        }
                    
                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(2*dely);                       
                    break;  

                    case 'R' :
                        topV = V[thread_idx-Nx];
                        btmV = V[thread_idx + Nx]; 

                        if ((thisV.BType == '0') || (thisV.BType == '2')){
                        /*Dirichlet Condition*/
                        leftV = U[thread_idx-1];                       
                        rightV =  2*thisV.UValue - leftV; 
                        } else {
                        rightV = 0;                       
                        leftV =  0;  
                        }
                    
                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(2*dely);    
                       
                    break;

                }

            case '2':
                switch (thisV.BSide){       
                    case 'W':
                    /*Top left*/

                    btmV = V[thread_idx + Nx]; 
                    rightV = U[thread_idx + 1]; 
                    topV = 2*thisV.VValue -  btmV; 
                    leftV = 2*thisV.UValue -  rightV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;

                    case 'D':
                    /*Top right*/
                    btmV = V[thread_idx + Nx]; 
                    leftV = U[thread_idx - 1]; 
                    topV = 2*thisV.VValue - btmV; 
                    rightV = 2*thisV.UValue -  leftV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;

                    case 'Z':
                    /*Bottom left*/

                    topV = V[thread_idx - Nx]; 
                    rightV = U[thread_idx + 1]; 
                    btmV = 2*thisV.VValue -  topV; 
                    leftV = 2*thisV.UValue -  rightV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;

                    case 'X':
                    /*Bottom Right*/

                    topV = V[thread_idx - Nx]; 
                    leftV = U[thread_idx - 1]; 
                    btmV = 2*thisV.VValue -  topV; 
                    rightV = 2*thisV.UValue - leftV; 

                    P_RHS[thread_idx] = (rightV - leftV)/(4*delx) + (topV - btmV)/(4*dely);    
                    break;
 
                    break; 
                }

        }

    }

}

__global__
void velPressureCorrection (double*P, double* U, double* V,const vertex* Domain, 
        int rows, int Nx, int Ny,double rho, double delT ){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    vertex thisV = Domain[thread_idx]; 
    double delx = thisV.delx;
    double dely = thisV.dely; 
    double constTerm = delT/rho *delx*dely; 

    double right, left, top, btm; 

    if (thread_idx < rows){

        switch (thisV.VType){
            case '0':
            /*Interior Vertex*/
            right = P[thread_idx + 1];
            left = P[thread_idx - 1];
            top = P[thread_idx - Nx];
            btm = P[thread_idx + Nx];

            U[thread_idx] = U[thread_idx] - ( right - left)/(2*delx)*constTerm;
            V[thread_idx] = V[thread_idx] - (top-btm)/(2*dely)*constTerm;
            break; 

            case '1':
            /*Edge*/
            switch (thisV.BSide){

                case 'T':
                    /*Top Edge*/
                    right = P[thread_idx + 1];
                    left = P[thread_idx - 1];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        top = 0;
                        btm = 0;
                    } else{
                        top = P[thread_idx-Nx];                       
                        btm =  2*thisV.PValue - top; 
                    }

                    U[thread_idx] = U[thread_idx] - ( right - left)/(2*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(4*dely)*constTerm;
                    break; 

                case 'B':
                    /*Btm Edge*/
                    right = P[thread_idx + 1];
                    left = P[thread_idx - 1];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        top = 0;
                        btm = 0;
                    } else{
                        btm = P[thread_idx+Nx];                       
                        top =  2*thisV.PValue - btm; 
                    }

                    U[thread_idx] = U[thread_idx] - ( right - left)/(2*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(4*dely)*constTerm;
                    break;

                case 'R':
                    /*Right Edge*/
                    top = P[thread_idx - Nx];
                    btm = P[thread_idx + Nx];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        right = 0;
                        left = 0;
                    } else{
                        left = P[thread_idx-1];                       
                        right =  2*thisV.PValue - left; 
                    } 

                    U[thread_idx] = U[thread_idx] - ( right - left)/(4*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(2*dely)*constTerm;
                    break;

                case 'L':
                    /*Left Edge*/
                    top = P[thread_idx - Nx];
                    btm = P[thread_idx + Nx];

                    if((thisV.VType == '0')||(thisV.VType == '1')){
                        right = 0;
                        left = 0;
                    } else{
                        right = P[thread_idx+1];                       
                        right =  2*thisV.PValue - right; 
                    } 

                    U[thread_idx] = U[thread_idx] - ( right - left)/(4*delx)*constTerm;
                    V[thread_idx] = V[thread_idx] - (top-btm)/(2*dely)*constTerm;
                    break;
            }


            break; 

            case '2':
                U[thread_idx] = U[thread_idx] ;
                V[thread_idx] = V[thread_idx] ;
                break; 

        }
    }
}

void Solve (int Lx, int Ly, int Nx,int Ny){

    int rows  = Ny*Nx; 
    vertex* DomainHost = (vertex*) malloc (sizeof(vertex)*rows);
    vertex* Domain = NULL; 
    double* Ulow, * Uhigh, * Udiag; 
    double* Vlow, *Vhigh, *Vdiag; 
    double* U, *V, *P; 
    double* H_u_n_1, *H_v_n_1, *H_u_n, *H_v_n; 
    double* D_u_n, *D_v_n;
    double* RHS_u_n, *RHS_v_n;
    double *P_rowPtr, *P_colPtr, *P_val, *P_RHS; 
  

    double* I = (double *)malloc(sizeof(double)*(rows));
    double *J = (double *)malloc(sizeof(double)*rows);
    double *val_host = (double *)malloc(sizeof(double) * rows );
    double *RHS = (double *)malloc(sizeof(double) * rows);
    double *Soln = (double *)malloc(sizeof(double) * rows);
    double*temp = (double *)malloc(sizeof(double) * 5*rows);

    double mu = 1;
    double rho = 1;
    double delT = 1.0;

    cublasHandle_t handleBlas = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handleBlas);
    checkCudaErrors (cublasStatus);


    checkCudaErrors(cudaMalloc((vertex **)&Domain, rows *sizeof(vertex)));

    checkCudaErrors(cudaMalloc((double **)&Ulow, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Uhigh, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Udiag, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&Vlow, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Vhigh, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Vdiag, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&U, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&V, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&H_u_n_1, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&H_v_n_1, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&H_u_n, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&H_v_n, rows *sizeof(double)));
    
    checkCudaErrors(cudaMalloc((double **)&D_u_n, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&D_v_n, rows *sizeof(double)));


    checkCudaErrors(cudaMalloc((double **)&RHS_u_n, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&RHS_v_n, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&P_rowPtr, (rows+1) *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P_colPtr, 5*rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P_val, 5*rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P_RHS, 5*rows *sizeof(double)));




    int blockWidth = 128;
    int nBlocks = rows/blockWidth + 1; 

    Mesh <<<nBlocks,blockWidth>>> (Lx,Ly,Ny,Nx, Domain);
    
    getPCoeffs<<<nBlocks,blockWidth>>>(P_rowPtr, P_colPtr, P_val,
             Domain,  rows, Nx, Ny, rho, delT);

    checkCudaErrors (cudaMemcpy( temp, P_colPtr,5*(rows)*sizeof(double), 
        cudaMemcpyDeviceToHost));

    printf ("\n Col Pointer\n");
    for (int i = 0; i < 5*rows; i++){

        printf("%f\n", temp[i]);
    }

    checkCudaErrors (cudaMemcpy( temp, P_val,5*(rows)*sizeof(double), 
    cudaMemcpyDeviceToHost));

    printf ("\n Val Pointer\n");
    for (int i = 0; i < 5*rows; i++){

        printf("%f\n", temp[i]);
    }



    genXCoeffs <<<nBlocks,blockWidth>>> (Ulow,Uhigh,Udiag,Domain, rows, mu, rho, delT); 
    genYCoeffs <<<nBlocks,blockWidth>>> (Vlow,Vhigh,Vdiag,Domain, rows, mu, rho, delT); 

    genRHS( RHS, rows, Nx); 
    checkCudaErrors (cudaMemcpy( U, RHS,(rows)*sizeof(double), 
        cudaMemcpyHostToDevice));

    printf ("\nInitial Mesh\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }



    genRHS2( RHS, rows, Nx); 
    checkCudaErrors (cudaMemcpy( V, RHS,(rows)*sizeof(double), 
        cudaMemcpyHostToDevice));

    printf ("\nInitial Mesh\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }


    calcH <<<nBlocks,blockWidth>>> (U,V,U, H_u_n, Domain, rows, Nx);

    // checkCudaErrors (cudaMemcpy( DomainHost, Domain,(rows)*sizeof(vertex), 
    //         cudaMemcpyDeviceToHost));
    // checkCudaErrors (cudaMemcpy( I, Ulow,(rows)*sizeof(double), 
    //     cudaMemcpyDeviceToHost));
    // checkCudaErrors (cudaMemcpy( J, Uhigh,(rows)*sizeof(double), 
    //         cudaMemcpyDeviceToHost));
    // checkCudaErrors (cudaMemcpy( val_host, Udiag,(rows)*sizeof(double), 
    //         cudaMemcpyDeviceToHost));

    checkCudaErrors (cudaMemcpy( RHS, H_u_n,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\nInitial Mesh\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

    calcH <<<nBlocks,blockWidth>>> (U,V,V, H_v_n, Domain, rows, Nx);

    checkCudaErrors (cudaMemcpy( RHS, H_v_n,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\nInitial Mesh\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

   calcD <<<nBlocks,blockWidth>>> (U,D_u_n, Domain, rows, Nx, 'U', delT, mu,
   rho );

    checkCudaErrors (cudaMemcpy( RHS, D_u_n,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\nInitial Mesh\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

    calcD <<<nBlocks,blockWidth>>> (V,D_v_n, Domain, rows, Nx, 'V', delT, mu, rho);

    checkCudaErrors (cudaMemcpy( RHS, D_v_n,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\nInitial Mesh\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

    updateRHS (handleBlas, H_u_n_1,  H_v_n_1,
      H_u_n,  H_v_n,  D_u_n, D_v_n, 
      RHS_u_n, RHS_v_n, delT, mu, rho, 
      rows); 

    checkCudaErrors (cudaMemcpy( RHS, RHS_u_n,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\nRHS_u\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

    TriDiagSolve(Ulow , Udiag, Uhigh, RHS_u_n, rows);

    checkCudaErrors(cublasDcopy(handleBlas,(rows), RHS_u_n, 1, U, 1));


    /* 3 - Transfer data from host to device*/
    checkCudaErrors (cudaMemcpy(J, Uhigh,  rows *sizeof(double), 
            cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy(val_host, Udiag, rows *sizeof(double), 
            cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy(I, Ulow, (rows)*sizeof(double),
             cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy( Soln, U,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("U: \n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", Soln[i]);
    }

    printOutput (I, J, val_host, RHS,Soln, rows);

    XtoY<<<nBlocks, blockWidth>>> ( RHS_v_n, Ny, Nx, V);

    checkCudaErrors (cudaMemcpy( RHS, V,(rows)*sizeof(double), 
        cudaMemcpyDeviceToHost));
    
    /*Need th re-center the coefficient matrices as well*/

    TriDiagSolve(Vlow , Vdiag, Vhigh, V, rows);

    checkCudaErrors (cudaMemcpy(J, Vhigh,  rows *sizeof(double), 
            cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy(val_host, Vdiag, rows *sizeof(double), 
            cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy(I, Vlow, (rows)*sizeof(double),
             cudaMemcpyDeviceToHost ));
    checkCudaErrors (cudaMemcpy( Soln, U,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printOutput (I, J, val_host, RHS,Soln, rows);

    
    YtoX<<<nBlocks, blockWidth>>> ( V, Ny, Nx, RHS_v_n);

    checkCudaErrors(cublasDcopy(handleBlas,(rows), RHS_v_n, 1, V, 1));

    checkCudaErrors (cudaMemcpy( RHS, V,(rows)*sizeof(double), 
            cudaMemcpyDeviceToHost));

    printf ("\n Final V\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }


    update_PRHS <<<nBlocks, blockWidth>>> ( P_RHS, U, V, Domain, rows, Nx, Ny,
                 rho); 

    // checkCudaErrors (cudaMemcpy( RHS, RHS_v_n,(rows)*sizeof(double), 
    //         cudaMemcpyDeviceToHost));

    // printf ("\nInitial Mesh\n");
    // for (int i = 0; i < rows; i++){

    //     printf("%f\n", RHS[i]);
    // }


    // printOutput (I,J,val_host, RHS, Soln, rows);

    // genYCoeffs <<<nBlocks,blockWidth>>> (Vlow,Vhigh,Vdiag,Domain, rows, mu, rho); 


    
    // checkCudaErrors (cudaMemcpy( I, Vlow,(rows)*sizeof(double), 
    //     cudaMemcpyDeviceToHost));
    // checkCudaErrors (cudaMemcpy( J, Vhigh,(rows)*sizeof(double), 
    //         cudaMemcpyDeviceToHost));
    // checkCudaErrors (cudaMemcpy( val_host, Vdiag,(rows)*sizeof(double), 
    //         cudaMemcpyDeviceToHost));

    // printOutput (I,J,val_host, RHS, Soln, rows);

    checkCudaErrors (cudaMemcpy( RHS, P_RHS,(rows)*sizeof(double), 
        cudaMemcpyDeviceToHost));

    printf ("\n Final P RHS\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

    velPressureCorrection <<<nBlocks, blockWidth>>>(P_RHS, U, V, Domain, 
         rows,  Nx, Ny, rho,  delT ); 

    checkCudaErrors (cudaMemcpy( RHS, U,(rows)*sizeof(double), 
        cudaMemcpyDeviceToHost));

    printf ("\n Pressure Corrected U\n");
    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }

    checkCudaErrors (cudaMemcpy( RHS, V,(rows)*sizeof(double), 
        cudaMemcpyDeviceToHost));

    printf ("\n Pressure Corrected V\n");

    for (int i = 0; i < rows; i++){

        printf("%f\n", RHS[i]);
    }


}

    
 
    
/* Starting point of the program
* Creates a linear system of a problem size from cmd prompts
* Solves the system using cuSparse
* Adds two vectors of problemSize using cuBlas
* 
*/

int main(int argc, char** argv){
    float time; 
    int Ny = 5;
    int Nx = 5;

    if (argc >= 2) {
            Ny = atoi(argv[1]);
            Nx = atoi(argv[2]);
    }

    // validate command line arguments
    if (Ny< 3) {
        
        Ny = 3;
        printf("Warning: Problem size can't be less than 3\n");
        printf("The total number of threads will be modified  to 3\n");
    }


    Solve(1,2,Ny,Nx);

    printf ("\n\nSolving  linear  equations using cuSparse library\n"); 
    //time = callDiff( 1,Nx, Ny);
    printf ("The time taken for linear solve is \n");
    //printf ("%3.1f mus", time );

    // printf ("\n\nAdding two vectors using cuBlas library\n");
    // time = vectorAdd (rows);  
    // printf ("The time taken for vector add is \n");
    // printf ("%3.1f mus", time );
}
