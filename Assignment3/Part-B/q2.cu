// Includes
#include <stdio.h>
/*
 * I reused code from provided files for profiling and tracking total runtime.
 *
*/

# ifndef TIMER_H
# define TIMER_H

/* initialize a timer, this must be done before you can use the timer! */
void initialize_timer ( void );

/* clear the stored values of a timer */
void reset_timer ( void );

/* start the timer */
void start_timer ( void );

/* stop the timer */
void stop_timer ( void );

/* return the elapsed time in seconds, returns -1.0 on error */
double elapsed_time ( void );

# endif /* TIMER_H */

// Variables for host and device arrays.
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C; 

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

__global__
void add(int n, float*sum, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    sum[i] = x[i] + y[i];
}

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    int k;
    int numberOfThreadsPerBlock;
    int numberOfBlocks;
    int N;

        // Parse arguments.
    if(argc != 4){
        printf("Usage: %s K numberOfThreadsPerBlock numberOfBlocks\n", argv[0]);
        printf("K is the length of the two arrays in millions.\n");
        printf("numberOfThreadsPerBlock is the total number of threads per block used to add the two arrays.\n");
        printf("numberOfBlocks is the total number of blocks used to add the two arrays.\n");
        exit(0);
    } else {
        sscanf(argv[1], "%d", &k);
        sscanf(argv[2], "%d", &numberOfThreadsPerBlock);
        sscanf(argv[3], "%d", &numberOfBlocks);
    } 
    
   
    N = k * 1000000;
    size_t size = N * sizeof(float);

    // Initialize shape for grid and blocks
    dim3 dimGrid(numberOfBlocks);                    
    dim3 dimBlock(numberOfThreadsPerBlock);                 

    // Allocate input arrays h_A, h_B, and h_C in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);

    // Allocate arrays d_A, d_B, and d_C in device memory.
    cudaError_t error;
    error = cudaMalloc((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize input arrays h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }

    // Copy input arrays h_A and h_B to device arrays d_A and d_B
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    // Warm up the kernel
    add<<<dimGrid, dimBlock>>>(N, d_C, d_A, d_B);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    add<<<dimGrid, dimBlock>>>(N, d_C, d_A, d_B);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

    // Compute floating point operations per second.
    int nFlops = N;
    double nFlopsPerSec = nFlops/time;
    double nGFlopsPerSec = nFlopsPerSec*1e-9;

    // Compute transfer rates.
    int nBytes = 3*4*N; // 2N words in, 1N word out
    double nBytesPerSec = nBytes/time;
    double nGBytesPerSec = nBytesPerSec*1e-9;

    // Report timing data.
    printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
             time, nGFlopsPerSec, nGBytesPerSec);
     
    // Copy result from device memory to host memory
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);

    // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");

    // Clean up and exit.
    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}

# define TIMER_C

# include <stdio.h>
# include <sys/time.h>


static double start, stop;        /* store the times locally */
static int start_flag, stop_flag; /* flag timer use */


void initialize_timer ( void )
{
    start = 0.0;
    stop  = 0.0;

    start_flag = 0;
    stop_flag  = 0;
}


void reset_timer ( void )
{
    initialize_timer();
}


void start_timer ( void )
{
    struct timeval time;

    if ( start_flag )
        fprintf( stderr, "WARNING: timer already started!\n" );

    start_flag = 1;

    if ( gettimeofday( &time, NULL ) < 0 )
        perror( "start_timer,gettimeofday" );

    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
}


void stop_timer ( void )
{
    struct timeval time;

    if ( !start_flag )
        fprintf( stderr, "WARNING: timer not started!\n" );

    if ( stop_flag )
        fprintf( stderr, "WARNING: timer already stopped!\n" );

    stop_flag = 1;

    if ( gettimeofday( &time, NULL ) < 0 )
        perror( "stop_timer,gettimeofday" );

    stop = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
}


double elapsed_time ( void )
{
    if ( !start_flag || !stop_flag )
        return (-1.0);

    return (stop-start);
}
