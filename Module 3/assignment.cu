/* Module 3 assignment submission
@ Nandan Joshi
* This module defines 4 functions - addition, subtraction, multiplication and modulo
* The functions are executed on a GPU and output pronted to console
*/

#include <stdio.h>
#include <time.h>
#include <math.h>
#include "accessory.h"


/* CUDA Kernel: addition
*  Takes two arrays and stores the element-wise addition in a third
*  array1, array2 : input arrays
* array3: output array
*/ 
__global__
void addition (int * array1, int* array2, int*array3){
    const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    array3[thread_idx] = array1[thread_idx] + array2[thread_idx]; 
    
}

/* CUDA Kernel: subtraction
*  Takes two arrays and stores the element-wise subtraction in a third
*  array1, array2 : input arrays
* array3: output array
*/ 

__global__
void subtraction (int * array1, int* array2, int*array3){
    const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    array3[thread_idx] = array1[thread_idx] - array2[thread_idx]; 
}

/* CUDA Kernel: multiplication
*  Takes two arrays and stores the element-wise multiplication in a third
*  array1, array2 : input arrays
* array3: output array
*/ 

__global__
void multiplication (int * array1, int* array2, int*array3){
    const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    array3[thread_idx] = array1[thread_idx] * array2[thread_idx]; 
}

/* CUDA Kernel: modulo
*  Takes two arrays and stores the element-wise mumodulus ltiplication in a third
*  array1 : input array containing dividends
*  array2 : Input array containing divisor
*  array3: output array
*/ 

__global__
void modulo (int * array1, int* array2, int*array3){
    const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
    array3[thread_idx] = array1[thread_idx] % array2[thread_idx]; 
}


/* Function: mainGPUexec
 * Generates two input arrays and performs add, subtract, multiply and modulo...
 *... operations on a GPU
 * totalThreads is the total no of GPU threads that would be used
 * blockSize is the number of threads per GPU block
 * printOutputFlag decides if the program output is written to console
 */

void mainGPUExec ( int totalThreads, int blockSize, int printOutputflag){


    int *inputArray1 =  (int*) malloc(totalThreads * sizeof(int));
    int *inputArray2 = (int*) malloc(totalThreads * sizeof(int));

    /*Initialize input arrays*/

    for (int i=0; i< totalThreads; i++){
        inputArray1[i] = i; 
    }

    for (int i=0; i< totalThreads; i++) {
        //Rnadom number between 1-3
        inputArray2[i] = (rand()%3 + 1);
    }


    //Declare output arrays
    int *addArray =  (int*) malloc(totalThreads * sizeof(int));
    int *subArray = (int*) malloc(totalThreads * sizeof(int));
    int *mulArray =  (int*) malloc(totalThreads * sizeof(int));
    int *modArray = (int*) malloc(totalThreads * sizeof(int));

    ///Declare pointers for GPU-based params
    int * gpu_inputArray1;
    int * gpu_inputArray2;
    int* gpu_addArray;
    int * gpu_subArray;
    int * gpu_mulArray; 
    int * gpu_modArray; 

    //Allocate arrays on GPU
    cudaMalloc((int **)&gpu_inputArray1, totalThreads * sizeof(int));
    cudaMalloc((int **)&gpu_inputArray2, totalThreads * sizeof(int));
    cudaMalloc((int **)&gpu_addArray, totalThreads * sizeof(int));
    cudaMalloc((void **)&gpu_subArray, totalThreads * sizeof(int));
    cudaMalloc((void **)&gpu_mulArray, totalThreads * sizeof(int));
    cudaMalloc((void **)&gpu_modArray, totalThreads * sizeof(int));


    // Copy data to GPU memory
    cudaMemcpy (gpu_inputArray1, inputArray1, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice); 

    cudaMemcpy (gpu_inputArray2, inputArray2, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice); 

    cudaMemcpy (gpu_addArray, addArray, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice);

    cudaMemcpy (gpu_subArray, subArray, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice);  

    cudaMemcpy (gpu_mulArray, mulArray, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice); 

    cudaMemcpy (gpu_modArray, modArray, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice); 


    /*Execute kernels*/  
    addition<<<totalThreads/blockSize, blockSize>>> (gpu_inputArray1, 
            gpu_inputArray2, gpu_addArray); 

    subtraction<<<totalThreads/blockSize, blockSize>>> (gpu_inputArray1, 
            gpu_inputArray2, gpu_subArray); 

    multiplication<<<totalThreads/blockSize, blockSize>>> (gpu_inputArray1, 
            gpu_inputArray2, gpu_mulArray); 

    modulo<<<totalThreads/blockSize, blockSize>>> (gpu_inputArray1, 
            gpu_inputArray2, gpu_modArray); 



    /*Copy memory back to host*/
    cudaMemcpy (addArray, gpu_addArray,totalThreads * sizeof(int), 
            cudaMemcpyDeviceToHost); 
    cudaMemcpy (subArray, gpu_subArray, totalThreads * sizeof(int), 
            cudaMemcpyDeviceToHost);  
    cudaMemcpy (mulArray, gpu_mulArray, totalThreads * sizeof(int), 
            cudaMemcpyDeviceToHost); 
    cudaMemcpy (modArray, gpu_modArray, totalThreads * sizeof(int), 
            cudaMemcpyDeviceToHost); 

    /*Free up the GPU memory*/
    cudaFree(gpu_inputArray1);
    cudaFree(gpu_inputArray2);
    cudaFree(gpu_addArray);
    cudaFree(gpu_subArray);
    cudaFree(gpu_mulArray);
    cudaFree(gpu_modArray);

    //print output
    if (printOutputflag == 1){
        printToConsole ( inputArray1, inputArray2, addArray, subArray, 
                         mulArray, modArray, totalThreads); 
    }

}

/* Function: testDifferentSizes()
*  Tries out different combinations of thread and block sizes
*/

void testDifferentSizes(){

    int threadSize [5] = {1024, 512, 256, 128, 64};
    int nBlocks [5] = {1,2,3,4,5}; 

    for (int i = 0; i < 5; i++){

        for (int j = 0; j<5; j++) {
            printf ("No of threads : %d\n", threadSize[i] );
            printf ("No of blocks : %d\n", nBlocks [j] );
            mainGPUExec ( threadSize[i] * nBlocks [j], threadSize[i], 0 ); 

        }


    }
}


/* Main function to interact with the console
*  Takes command line arguments for GPU computation
*  First argument is the total number of threads required
*  Second argument is the number of blocks required
*  If no arguments are specified, tests the program over a range of sizes
*/

int main(int argc, char** argv){

    // read command line arguments

    if (argc<2){
        /*No arguments specified*/
        testDifferentSizes();
         
    } else{

        int totalThreads = 32;
        int blockSize = 16;
        int numBlocks = totalThreads/blockSize;

        if (argc >= 2) {
            totalThreads = atoi(argv[1]);
        }
        if (argc >= 3) {
            blockSize = atoi(argv[2]);
        }

        // validate command line arguments
        if (totalThreads % blockSize != 0) {
            ++numBlocks;
            totalThreads = numBlocks*blockSize;
            
            printf("Warning: Total thread count is not evenly divisible by the block size\n");
            printf("The total number of threads will be rounded up to %d\n", totalThreads);
        }

        /*Call the function to generate inputs and calculate all output arrays*/
        mainGPUExec ( totalThreads, blockSize, 1); 


    }
 
}






