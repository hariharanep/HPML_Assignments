#include <cstdlib>
#include <iostream>
#include <cmath>
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

// Variables for host vectors
float* h_A; 
float* h_B; 
float* h_C;

void Cleanup();

int main(int argc, char* argv[]) {
  int k;

  if(argc != 2){
    printf("Usage: %s K\n", argv[0]);
    printf("K is the length of the arrays in millions.\n");
    exit(0);
  } else {
    sscanf(argv[1], "%d", &k);
  }  

  long N = k * 1000000;
  size_t size = N * sizeof(float);
  printf("Total size of each array: %d\n", N * sizeof(float));

  // Allocate space for host arrays
  h_A = (float*)malloc(size);
  if (h_A == 0) Cleanup();
  h_B = (float*)malloc(size);
  if (h_B == 0) Cleanup();
  h_C = (float*)malloc(size);
  if (h_C == 0) Cleanup();

  // Initialize host array A, host array B
  int i;
  for(i=0; i<N; ++i){
    h_A[i] = (float)i;
    h_B[i] = (float)(N-i);   
  }

  // Warm up CPU
  for(i=0; i<N; ++i){
    h_C[i] = h_A[i] + h_B[i];
  }

  // Initialize timer  
  initialize_timer();
  start_timer();

  // execute 1D array addition
  for(i=0; i<N; ++i){
    h_C[i] = h_A[i] + h_B[i];
  }

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

  // Verify & report result
  for (i = 0; i < N; ++i) {
    float val = h_C[i];
    if (fabs(val - N) > 1e-5)
      break;
  }
  printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
  
  // Clean up and exit.
  Cleanup();

}

void Cleanup() {
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
           
  fflush( stdout);
  fflush( stderr);

  exit(0);
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

