#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include<fstream.h>



void writeOutOutFile (double* resArray, int rows, int nPrintedSteps, char VarId){
    FILE *fp1; 

    char name[] = " Output.csv"; 
    char fullName[25]; 
    fullName [0] = varId; 
    fullName[1] = "\0"; 
    strcat (fullName, name);     

    fp1 = fopen (fullName, "w"); 

    if (fp1 == NULL)
    {
        printf("Error while opening the file.\n");
    }

    for (int i = 0; i< nPrintedSteps; i++){
        for (int (j = 0; j<rowsl j++)){
            fprintf ("%0.2f", resArray[j + nPrintedSteps]); 
            fprintf (","); 

        }
    fprintf ("\n");
    }

    fclose (fp1); 
}