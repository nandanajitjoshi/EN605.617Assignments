#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <iostream>


#include <chrono>
using namespace std::chrono;

/* Funcion : printOutput
* Prints output of 4 operations
*/
void printOutput (thrust::device_vector<int> add, thrust::device_vector<int> sub, 
      thrust::device_vector<int> mul, thrust::device_vector<int> mod  ){


        std::cout << "\nThe results of addition : \n"  << std::endl;

        for(int i = 0; i < add.size(); i++){
       
            std::cout << "Add Array[" << i << "] = " << add[i] << std::endl;
        } 

        std::cout << "\nThe results of subtraction : \n"  << std::endl;


        for(int i = 0; i < sub.size(); i++){
       
            std::cout << "Sub Array[" << i << "] = " << sub[i] << std::endl;
        }     


        std::cout << "\nThe results of multiplication : \n"  << std::endl;

        for(int i = 0; i < mul.size(); i++){
       
            std::cout << "Mul Array[" << i << "] = " << mul[i] << std::endl;
        }  

        std::cout << "\nThe results of modulo : \n"  << std::endl;


        for(int i = 0; i < mod.size(); i++){
       
            std::cout << "Mod Array[" << i << "] = " << mod[i] << std::endl;
        }  
}

/* Funcion : performOperations
* Performs 4 operations - add, subtract, multiply and modulo on two input arrays
* Returns the time taken for execution
*/

void  performOperations( thrust::device_vector<int> vec1, 
    thrust::device_vector<int> vec2 ){

    /*Initialize output vectors*/
    thrust::device_vector<int> add = vec2;
    thrust::device_vector<int> sub = vec2;
    thrust::device_vector<int> mul = vec2;
    thrust::device_vector<int> mod = vec2;


    //Record time
    auto start = high_resolution_clock::now();


    /*Perform operations*/
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), add.begin(),
            thrust::plus<int>());                  //add
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), sub.begin(), 
            thrust::minus<int>());                 //subtract
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), mul.begin(), 
            thrust::multiplies<int>());            //multiply
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), mod.begin(), 
            thrust::modulus<int>());               //modulo


   /*Record time*/
    auto stop = high_resolution_clock::now();

    /*Print output*/      
    printOutput(add, sub, mul, mod);
    

    auto duration = duration_cast<microseconds>(stop - start); 
    /*Print time*/
    std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;

    //return time;

}

/* Funcion : Main
* Entry point of program
* Initializes two arays of user-defined size
  input: argv lists the input array size
*/

int main(int argc, char** argv)
{
    int totalThreads = (1 << 5);
    float time; 
    if (argc >= 2) {
        /*Input size from console*/
        totalThreads = atoi(argv[1]);
    }

    // Initialize host vectors
    thrust::host_vector<int> H1(totalThreads);
    thrust::host_vector<int> H2(totalThreads);

    //Fill host vectors
    thrust::sequence(H1.begin(), H2.end()); //H[1] = , H[i] = 2
    thrust::fill(H2.begin(),H2.end(),2); //Each element is 2


    // Copy host_vector H to device_vector D
    thrust::device_vector<int> input1 = H1;
    thrust::device_vector<int> input2 = H2;

    //Print input vectors

    std::cout << "The furst array : \n"  << std::endl;

    for(int i = 0; i < H1.size(); i++){

        std::cout << "Input 1[" << i << "] = " << H1[i] << std::endl;
    } 

    std::cout << "\nThe second array : \n"  << std::endl;


    for(int i = 0; i < H2.size(); i++){

        std::cout << "Input 2[" << i << "] = " << H2[i] << std::endl;
    }  

    //Perform the mathematical operations 
    performOperations (input1,input2);


    // H and D are automatically destroyed when the function returns
    return 0;
}
