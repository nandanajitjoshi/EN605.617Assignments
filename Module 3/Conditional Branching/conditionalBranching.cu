/* Module 3 assignment submission
@ Nandan Joshi
* This module dshows the effect of conditional branching with three functions
* First function has branches for each alternate thread
* Second kernel has only one branch at a warp half-boundary
* Third kernel does not have any branches and is passed processed data
*/


#include <stdio.h>
#include <time.h>
#include <math.h>

/* Kernel : insideBranching1()
* Doubles all the even positions and triples all odd positions
* array1 is the input array, which is modified by the kernel
*/

__global__
void insideBranching1 (int * array1){

	const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 


    if (thread_idx %2 == 0){
        // Double even positions
        array1 [thread_idx] = 2*array1 [thread_idx];
    } else if (thread_idx %2 == 1) {
        //Triples Odd positions
        array1 [thread_idx] = 3*array1 [thread_idx];
    }
	
}

/* Kernel : insideBranching2()
* Doubles all  numbers in first half and triples all in the second half of array
* array1 is the input array, which is modified by the kernel
* totalThreads is the total number of threads created, and is also the array size
*/

__global__
void insideBranching2 (int * array1, int totalThreads){

    const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (thread_idx <totalThreads/2){
        array1 [thread_idx] = 2*array1 [thread_idx];
    } else {
        array1 [thread_idx] = 3*array1 [thread_idx];
    }
	
}


/* Kernel : outsideBranching1()
* Doubles all  numbers in first array, and triples all in the second
* array1, array2 are is the input arrays, which is modified by the kernel
*/
__global__
void outsideBranching1 (int * evenArray, int* oddArray){

    const unsigned int  thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; 


   
    evenArray [thread_idx] = 2*evenArray [thread_idx];
    oddArray [thread_idx] = 3*oddArray [thread_idx];
}

/* Function : branchInsideKernel()
* Creates an array, to be passed to a CUDA kernel
* The CUDA kernal shows consitional branching at alternate threads*/

void branchInsideKernel ( int totalThreads, int blockSize, float* times){	


	int *inputArray1 =  (int*) malloc(totalThreads * sizeof(int));

    for (int i=0; i< totalThreads; i++){
        inputArray1[i] = i; 
    }

    /*Declare pointers for GPU-based params*/
	int * gpu_inputArray1;
	

	/*Allocate arrays on GPU*/
	cudaMalloc((int **)&gpu_inputArray1, totalThreads * sizeof(int));

    /*Create timer variables*/
    cudaEvent_t startMem, stopMem, startProc, stopProc;
    float timeMem, timeProc;
    cudaEventCreate (&startMem);
    cudaEventCreate (&stopMem);
    cudaEventRecord (startMem, 0);

    /*See effect of conditional branching within the kernel*/
	cudaMemcpy (gpu_inputArray1, inputArray1, totalThreads * sizeof(int), cudaMemcpyHostToDevice  ); 
	

    /*Create timer variables for just the kermal execution time*/
    cudaEventCreate (&startProc);
    cudaEventCreate (&stopProc);
    cudaEventRecord (startProc, 0);

	/*Execute kernel*/
	insideBranching1<<<totalThreads/blockSize, blockSize>>> (gpu_inputArray1);

    /* Time taken for just the kernel execution*/
    cudaEventRecord (stopProc, 0);
    cudaEventSynchronize (stopProc);
    cudaEventElapsedTime (&timeProc, startProc, stopProc);
    cudaEventDestroy (startProc);
    cudaEventDestroy (stopProc);    



	/*Copy memory back to host*/
	cudaMemcpy (inputArray1, gpu_inputArray1,totalThreads * sizeof(int), cudaMemcpyDeviceToHost  ); 

    /* Time taken for  kernel execution and memory transfer*/
    cudaEventRecord (stopMem, 0);
    cudaEventSynchronize (stopMem);
    cudaEventElapsedTime (&timeMem, startMem, stopMem);
    cudaEventDestroy (startMem);
    cudaEventDestroy (stopMem);   

    timeProc = timeProc * pow(10,3); 
    timeMem = timeMem * pow(10,3); 


	/*Free up the GPU memory*/
	cudaFree(gpu_inputArray1);

    free (inputArray1);

    times[0]= timeProc;
    times[1] = timeMem;  


}

/* Function : branchInsideModified()
* Creates  another array that has  even indices in first half and odd in the secind
* Passes the modified array to the kernel to utilize warp half-boundary branching 
*/


void branchInsideModified ( int totalThreads, int blockSize, float*times){	


	int *inputArray1 =  (int*) malloc(totalThreads * sizeof(int));
    int *intermediateArray = (int*) malloc(totalThreads * sizeof(int));


    for (int i=0; i< totalThreads; i++){
        inputArray1[i] = i; 
    }

    /*Condition the input array*/

    for (int i=0; i< totalThreads/2; i++){
        intermediateArray[i] = inputArray1[2*i]; 
    }

    for (int i=0; i< totalThreads/2; i++){
        intermediateArray[i+totalThreads/2] = inputArray1[(2*i+1)]; 
    }

    /*Declare pointers for GPU-based params*/

	int * gpu_inputArray1;
	

	/*Allocate arrays on GPU*/
	cudaMalloc((int **)&gpu_inputArray1, totalThreads * sizeof(int));
	

    cudaEvent_t startMem, stopMem, startProc, stopProc;
    float timeMem, timeProc;

    /*Create timer variables for kernel proceess and memory transfer*/
    cudaEventCreate (&startMem);
    cudaEventCreate (&stopMem);
    cudaEventRecord (startMem, 0);

	cudaMemcpy (gpu_inputArray1, intermediateArray, totalThreads * sizeof(int), 
            cudaMemcpyHostToDevice  ); 
	
	
    /*Create timer variables for kernel proceess alone*/
    cudaEventCreate (&startProc);
    cudaEventCreate (&stopProc);
    cudaEventRecord (startProc, 0);

	/*Execute kernel*/
	insideBranching2<<<totalThreads/blockSize, blockSize>>> (gpu_inputArray1, 
            totalThreads);

    /*Calculate time for kernel processing*/
    cudaEventRecord (stopProc, 0);
    cudaEventSynchronize (stopProc);
    cudaEventElapsedTime (&timeProc, startProc, stopProc);
    cudaEventDestroy (startProc);
    cudaEventDestroy (stopProc);    


	/*Copy memory back to host*/
	cudaMemcpy (intermediateArray, gpu_inputArray1,totalThreads * sizeof(int), 
            cudaMemcpyDeviceToHost  ); 
	
    /*Calculate time for kernel processing + memory transfer*/
    cudaEventRecord (stopMem, 0);
    cudaEventSynchronize (stopMem);
    cudaEventElapsedTime (&timeMem, startMem, stopMem);
    cudaEventDestroy (startMem);
    cudaEventDestroy (stopMem);  

    timeProc = timeProc * pow(10,3); 
    timeMem = timeMem * pow(10,3); 

	/*Free up the GPU memory*/
	cudaFree(gpu_inputArray1);


    free (inputArray1);
    free (intermediateArray);   

    times[0]= timeProc;
    times[1] = timeMem;  



}


/* Function : branchOutside()
* Creates an array, and then creates two new arrays of its ...
*...even and odd-numbered positions
* Even and odd arrays  passed to a kernel that operates on them seperately
*/

void branchOutsideKernel ( int totalThreads, int blockSize, float*times){	

    /*Allocate memory for arrays*/
	int *inputArray1 =  (int*) malloc(totalThreads * sizeof(int));
    int *evenArray = (int*) malloc(totalThreads/2 * sizeof(int));
    int *oddArray = (int*) malloc(totalThreads/2 * sizeof(int));

    /*Initialize input arrays*/

    for (int i=0; i< totalThreads; i++){
        inputArray1[i] = i; 
    }

    /*Create the even and odd arrays*/
    for (int i=0; i< totalThreads/2; i++){
        evenArray[i] = inputArray1[2*i]; 
    }

    for (int i=0; i< totalThreads/2; i++){
        oddArray[i] = inputArray1[(2*i+1)]; 
    }

    /*Declare pointers for GPU-based params*/
	int * gpu_evenArray1;
	int * gpu_oddArray;
	

	/*Allocate arrays on GPU*/
	cudaMalloc((int **)&gpu_evenArray1, totalThreads/2 * sizeof(int));
	cudaMalloc((int **)&gpu_oddArray, totalThreads/2 * sizeof(int));


    cudaEvent_t startMem, stopMem, startProc, stopProc;
    float timeMem, timeProc;


    /*Create timer variables for kernel proceess + memory transfer*/
    cudaEventCreate (&startMem);
    cudaEventCreate (&stopMem);
    cudaEventRecord (startMem, 0);

	cudaMemcpy (gpu_evenArray1, evenArray, totalThreads/2 * sizeof(int), 
            cudaMemcpyHostToDevice  ); 
	cudaMemcpy (gpu_oddArray,oddArray, totalThreads/2 * sizeof(int), 
            cudaMemcpyHostToDevice  ); 
	
    /*Create timer variables for kernel proceess alone*/
    cudaEventCreate (&startProc);
    cudaEventCreate (&stopProc);
    cudaEventRecord (startProc, 0);

    
    /*Execute kernel*/
	outsideBranching1<<< totalThreads/(2*blockSize), blockSize>>> (gpu_evenArray1,
            gpu_oddArray);

    /*Calculate time for kernel processing*/
    cudaEventRecord (stopProc, 0);
    cudaEventSynchronize (stopProc);
    cudaEventElapsedTime (&timeProc, startProc, stopProc);
    cudaEventDestroy (startProc);
    cudaEventDestroy (stopProc);    


	/*Copy memory back to host*/
	cudaMemcpy (evenArray, gpu_evenArray1,totalThreads/2 * sizeof(int), 
            cudaMemcpyDeviceToHost  ); 
	cudaMemcpy (oddArray, gpu_oddArray, totalThreads/2 * sizeof(int), 
            cudaMemcpyDeviceToHost  );  

    /*Calculate time for kernel processing + memory transfer*/
    cudaEventRecord (stopMem, 0);
    cudaEventSynchronize (stopMem);
    cudaEventElapsedTime (&timeMem, startMem, stopMem);
    cudaEventDestroy (startMem);
    cudaEventDestroy (stopMem);  

    timeProc = timeProc * pow(10,3); 
    timeMem = timeMem * pow(10,3); 

	/*Free up the GPU memory*/
	cudaFree(gpu_evenArray1);
	cudaFree(gpu_oddArray);

    free (evenArray);
    free (oddArray);

    
    times[0] = timeProc; 
    times[1] = timeMem; 
 
  


}



int main(int argc, char** argv)
{
	// read command line arguments
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

    float times[2] = {0.0,0.0}, 
    float avgTimes[2] = {0.0,0.0}; 
    int nTrials = 5; 

    for (int i=0; i<nTrials;i++ ){
        branchInsideKernel ( totalThreads, blockSize, times); 
        avgTimes [0] = (avgTimes[0] + times[0]);
        avgTimes [1] = (avgTimes[1] + times[1]);
    }

    printf("Time to run kernel with branching inside kernel:  %3.1f mus \n", 
            avgTimes[0]/nTrials);
    printf("Time including memory transfer:  %3.1f ns \n", avgTimes[1]/nTrials);

    avgTimes[0] = 0.0;
    avgTimes[1] = 0.0;

    for (int i=0; i<nTrials;i++ ){
        branchInsideModified (totalThreads, blockSize, times);
        avgTimes [0] = (avgTimes[0] + times[0]);
        avgTimes [1] = (avgTimes[1] + times[1]);
    }

    printf("Time to run kernel with branching inside kernel at half-warp:  %3.1f mus \n", 
            avgTimes[0]/nTrials);
    printf("Time including memory transfer:  %3.1f mus \n", avgTimes[1]/nTrials);

    avgTimes[0] = 0.0;
    avgTimes[1] = 0.0;
    
    for (int i=0; i<nTrials;i++ ){
        branchOutsideKernel (totalThreads, blockSize, times); 
        avgTimes [0] = (avgTimes[0] + times[0]);
        avgTimes [1] = (avgTimes[1] + times[1]);
    }

    printf("Time to run kernel with branching outside kernel:  %3.1f mus \n", avgTimes[0]/nTrials);
    printf("Time including memory transfer:  %3.1f mus \n", avgTimes[1]/nTrials);
    


}


 







