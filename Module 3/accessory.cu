#include <stdio.h>
#include <time.h>
#include <math.h>

/*Contains accesory functions to the main module*/


/* Function : printToConsole
*  Takes the input and output arrays and prints them to the console
* inputArray1, inputArray2, addArray, subArray, mulArray, modArray : input arrays
* totalThreads: Number of elements to be pronted
*/
void printToConsole( int* inputArray1, int* inputArray2, int* addArray, int* subArray, 
                    int* mulArray, int* modArray, int totalThreads) {


    /*Print  out the input arrays*/
    printf ("Input array 1: \n"); 
    printf ("\n-------------------------------\n"); 	
    for (int i=0; i< totalThreads; i++) {
         printf("%d \t", inputArray1[i]);    

        if ((i%15 == 0) && (i!=0)){
            /*Only 15 characters in one line*/
            printf ("\n");
        } 
    }

    printf ("\n-------------------------------\n"); 
    printf ("\nInput array 2: \n"); 
    printf ("\n-------------------------------\n"); 	

    for (int i=0; i< totalThreads; i++) {
         printf("%d \t", inputArray2[i]);    

        if ((i%15 == 0) && (i!=0)){
            /*Only 15 characters in one line*/
            printf ("\n");
        } 
    }

    printf ("\n-------------------------------\n"); 	

    /*Print  out the output arrays*/

    printf ("\nAdd array : \n"); 
    printf ("\n-------------------------------\n"); 	

    for (int i=0; i< totalThreads; i++) {
         printf("%d \t", addArray[i]); 

        if ((i%15 == 0) && (i!=0)){
            /*Only 15 characters in one line*/
            printf ("\n");
        }     
    }
    printf ("\n-------------------------------\n"); 	
    printf ("\n Subtract array : \n"); 
    printf ("\n-------------------------------\n"); 	

    for (int i=0; i< totalThreads; i++) {
         printf("%d \t", subArray[i]);    

        if ((i%15 == 0) && (i!=0)){
            /*Only 15 characters in one line*/
            printf ("\n");
        }     
    }

    printf ("\n-------------------------------\n"); 	
    printf ("\n Multiplication array : \n"); 
    printf ("\n-------------------------------\n"); 	

    for (int i=0; i< totalThreads; i++) {
         printf("%d \t", mulArray[i]); 

        if ((i%15 == 0) && (i!=0)){
            /*Only 15 characters in one line*/
            printf ("\n");
        }     
    }

    printf ("\n-------------------------------\n"); 	
    printf ("\n Modulo array : \n"); 
    printf ("\n-------------------------------\n"); 	

    for (int i=0; i< totalThreads; i++) {
         printf("%d \t", modArray[i]);  

        if ((i%15 == 0) && (i!=0)){
            /*Only 15 characters in one line*/
            printf ("\n");
        }     
    }
}



    
