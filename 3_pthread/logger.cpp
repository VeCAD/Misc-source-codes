/*Using back some of the functions from my previous work
which is mostly in C syntax
Adding pthreads for multithreaded calls
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "read_mle.cpp"
#include <pthread.h>

#define NUM_OF_THREADS 3

/********Main function**************/
int main(/*int argc, char **argv*/) {

//Framework choice- Loads MLE(Max Likelihood Estimates) data from ./train folder 1035 x 2304 float
//and each thread will print/save txt separately
//Functions are in read_mle.cpp file
double *traindata;
read_mle(&traindata);
printf("Naive Bayes Classifier training data loaded\nPress ENTER key to continue...\n");
while(getchar() != '\n');



//Multithreaded call of print_mle
struct print_mle_args args[NUM_OF_THREADS]; 
pthread_t threads[NUM_OF_THREADS];
int i; 

    //for each threaded function assign its individual data so we can run them in parallel
    for( i = 0; i < NUM_OF_THREADS; i++ ) 
    {
    args[i].train=traindata;
    args[i].tid = i;
    }
     
    //create pthreads
    for( i = 0; i < NUM_OF_THREADS; i++ ) 
    {
    pthread_create(&threads[i], NULL, print_mle, (void*)&args[i]);
    }
    
    //pthread join
    for( i = 0; i < NUM_OF_THREADS; i++ ) 
    {
    pthread_join(threads[i], NULL);
    }
        
free(traindata);

return 0;
}
