//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
//const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes seven arguments: result [5] (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[7],
                      float *a, float *b)
{
    int size = 50;
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * size, NULL, NULL);
    // memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                                sizeof(float) * size, NULL, NULL);

    // memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                                sizeof(float) * size, NULL, NULL);

    // memObjects[5] = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                             sizeof(float) * size, NULL, NULL);

    // memObjects[6] = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                             sizeof(float) * size, NULL, NULL);

    for (int i = 0; i<7; i++){
        if (memObjects[i] == NULL)
        {
            std::cerr << "Error creating memory objects." << std::endl;
            return false;
        }
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel[5], cl_mem memObjects[7])
{
    for (int i = 0; i < 7; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    for (int i = 0; i < 7; i++)
    {
        if (kernel[i] != 0)
            clReleaseKernel(kernel[i]);
    }
   
    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

/*Function printOutput
* Prints the input arrays and the results of the math operations
*/

void printOutput (int i, float* a, float* b, float * result,  
        int size, float duration ){
    
    // Print input buffer

    int lineReturn = 15;

    switch (i){

        case 0:
            std::cout << "The output of addition a+b : "<< std::endl;
            break; 

        case 1:
            std::cout << "The output of subtraction a-b : "<< std::endl;
            break; 

        case 2: 
             std::cout << "The output of multiplication a*b : "<< std::endl;
            break;  

        case 3: 
             std::cout << "The output of division a/b : "<< std::endl;
            break;  

        case 4: 
             std::cout << "The output of exponentiation b^a : "<< std::endl;
            break;  
    }

    // Output the result buffer
    for (int i = 0; i < size; i++)
    {
        std::cout << result[i] << " ";

        if ((i%lineReturn==0 && i!=0)){
            std::cout<<"\n"; 
        }
    }
    std::cout << std::endl;

    std::cout<< "The time for this kernel is : "<< duration <<std::endl;

}
/* Function executeFunctionsHost
* Host side function to call the kernel and transfer data back to host
* Inputs : context, commandQueue, program, device, errNum and memObjects are CL
* variables necessary for calling the kernels
* Input: size is the size of the arrays
* Input : a,b are the input arrays
*/


float executeFunctionsHost (int size, cl_context context, 
    cl_command_queue commandQueue, cl_program program, 
    cl_device_id device, cl_kernel kernel, cl_mem memObjects[3], 
    cl_int errNum, float*a, float*b, float* addResult
    ){

        /*Create events for timing*/
        cl_event event[8];  

        // Initialize the start- and end time for the event
        unsigned long start = 0;
        unsigned long end = 0;

        size_t globalWorkSize[1] = { size };
        size_t localWorkSize[1] = { 1 };

        // Queue the kernel up for execution across the array
        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                        globalWorkSize, localWorkSize,
                                        0, NULL, &event[0]);

        //Record the start time
        clGetEventProfilingInfo(event[0],CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong),&start,NULL);       



        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error queuing kernel for execution." << std::endl;
            //Cleanup(context, commandQueue, program, kernel, memObjects);
            //return 1;
        }

        // Read the output buffers back to the Host
        errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                    0, size * sizeof(float), addResult,
                                    1, &event[0], &event[1]);

        // errNum |= clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE,
        //                             0, size * sizeof(float), subResult,
        //                             2, &event[0], &event[2]);

        // errNum |= clEnqueueReadBuffer(commandQueue, memObjects[4], CL_TRUE,
        //                         0, size * sizeof(float), mulResult,
        //                         3, &event[0], &event[3]);

        // errNum |= clEnqueueReadBuffer(commandQueue, memObjects[5], CL_TRUE,
        //                     0, size * sizeof(float), divResult,
        //                    4, &event[0], &event[4]);

        // errNum |= clEnqueueReadBuffer(commandQueue, memObjects[6], CL_TRUE,
        //                 0, size * sizeof(float), powResult,
        //                 5, &event[0], &event[5]);

        /*Record the end time*/
        clGetEventProfilingInfo(event[1],CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong),&end,NULL);

        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error reading result buffer." << std::endl;
            //Cleanup(context, commandQueue, program, kernel, memObjects);
            //return 1;
        }


        /*Return the duration in nanoseconds*/
        return ((float)(end - start)/1E9); 

    }
///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    int ARRAY_SIZE = 50;

    if (argc >= 2) {
            ARRAY_SIZE = atoi(argv[1]);
    }

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel[5] = {0,0,0,0,0};
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum = 0;
    float duration; 

    const char* mathOperations [5] = {"add", "sub", "mul", "div", "power"};

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "HelloWorld.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    for (int i = 0; i<5; i++){
    // Create OpenCL kernel
        kernel[i] = clCreateKernel(program, mathOperations[i], NULL);
        if (kernel[i] == NULL)
        {
            std::cerr << "Failed to create kernel" << std::endl;
            Cleanup(context, commandQueue, program, kernel, memObjects);
            return 1;
        }
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float* result = (float *)malloc(sizeof(float)*ARRAY_SIZE);
    // float* subResult = (float *)malloc(sizeof(float)*ARRAY_SIZE);
    // float* mulResult = (float *)malloc(sizeof(float)*ARRAY_SIZE);
    // float* divResult = (float *)malloc(sizeof(float)*ARRAY_SIZE);
    // float* powResult = (float *)malloc(sizeof(float)*ARRAY_SIZE);
    float* a  = (float *)malloc(sizeof(float)*ARRAY_SIZE);
    float* b  = (float *)malloc(sizeof(float)*ARRAY_SIZE);

    /*Initialize the inputs*/
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    if (!CreateMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    for (int i=0; i<5; i++){

    // Set the kernel arguments (result, a, b)
        for (int j=0; j < 3; j++){
            errNum |= clSetKernelArg(kernel[i], j, sizeof(cl_mem), &memObjects[j]);
        }

        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error setting kernel arguments." << std::endl;
            Cleanup(context, commandQueue, program, kernel, memObjects);
            return 1;
        }

        /*Call the function that executes the kernels and returns time*/
        duration = executeFunctionsHost (ARRAY_SIZE,  context, 
        commandQueue, program, device,  kernel[i], memObjects, errNum,
        a,b, result); 

        /*Print the output*/
        printOutput ( i, a, b, result, ARRAY_SIZE, duration );

        errNum = 0;
    }

    /*Print the output*/
    // printOutput ( a, b, addResult, subResult,  mulResult,
    //                  divResult,  powResult, ARRAY_SIZE );


    std::cout<< "The time taken for the kernel execution "
    <<duration <<" secs"<< std::endl; 

    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);

    return 0;
}
