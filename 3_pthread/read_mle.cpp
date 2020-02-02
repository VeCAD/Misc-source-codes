///////////////////////////////////////////////////////////////Read MLE from train folder/////////////////////////////////////////////
void read_mle(double **traindata)
{

unsigned int feature_size = 9*256;	         //9 partitions x histogram bin size
unsigned int subject = 1035;                 //no. of subjects
unsigned int y,x;

*traindata = (double*)malloc(sizeof(double)*subject*feature_size);
memset(*traindata, 0, sizeof(double)*subject*feature_size);

//Load training features from file
    FILE *Train;
    Train = fopen("./train/ml_estimate.txt", "r");

    for (y=0; y < subject; y++)
    {
    for (x=0; x < feature_size; x++) 
    {
    if ( fscanf(Train, "%lf,", &(*traindata)[y*feature_size + x]) != 1 ) {printf ("Training data read error\n"); exit(1);}
    }
    }

    fclose(Train); 

return;
}


///////////////////////////////////////////////////////////////Logger Printer function/////////////////////////////////////////////

//for pthread need to wrap arguments in struct to pass to func
struct print_mle_args{
int tid;       //threadid
double *train;  //the data read from file
};


/********Logger/Printer**************/
//Each thread will save to different output file
void *print_mle(void *traindata)  
{
//these parameters must be same as training
unsigned int feature_size = 9*256;  //9 partitions x histogram bin size
unsigned int subject = 1035;           //no. of subjects

//required for pthread , transfer pointer
struct print_mle_args *args = (struct print_mle_args *)traindata;

//use fopen to save txt file
FILE *FW;
char open_label[50];
sprintf(open_label,"output_log_thread_%d.txt",args->tid);
FW = fopen(open_label,"w");
if(FW == NULL)
{
printf("Error opening file!\n");
exit(1);
}
//double *train = (double *) traindata; //transfer pointer
unsigned int y,x;
double *px = (double *)malloc(sizeof(double)*subject*feature_size);
memset(px, 0, sizeof(double)*subject*feature_size); //zero values

    for (y=0; y < subject; y++)
    {
    for (x=0; x < feature_size; x++) 
    {
    //Comment out if require either terminal printout or text file printout
    px[y*feature_size + x] = args->train[y*feature_size + x];           //Load output array
    printf("ThreadId=%d, %lf\n", args->tid, px[y*feature_size + x]);    //Print value to terminal with strings
    fprintf(FW, "ThreadId=%d, %lf\n",args->tid, px[y*feature_size + x]);//Save same output to text with fprintf
    }
    //fprintf(FW, "\n");
    }
    fclose(FW);

pthread_exit(0);
}

