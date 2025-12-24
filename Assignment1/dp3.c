#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl_cblas.h>

float bdp(long N, float *pA, float *pB) {
  float R = cblas_sdot(N, pA, 1, pB, 1);
  return R;
}

int main(int argc, char *argv[]) {
  long arrSize = atol(argv[1]);
  long repetitions = atol(argv[2]);
  
  float *arr1 = malloc(arrSize * sizeof(float));
  float *arr2 = malloc(arrSize * sizeof(float));
  for (long i = 0; i < arrSize; i++) {
    arr1[i] = 1.0;
    arr2[i] = 1.0;
  }
  long halfway = repetitions / 2;
  double total_time = 0;
  float temp = 0.0;
  for (int i = 0; i < repetitions; i++) {
    struct timespec start_time, end_time;
  
    if (clock_gettime(CLOCK_MONOTONIC, &start_time) == -1) {
      perror("clock_gettime");
      return 1;
    }

    temp = bdp(arrSize, arr1, arr2);
    
    if (clock_gettime(CLOCK_MONOTONIC, &end_time) == -1) {
      perror("clock_gettime");
      return 1;
    }

    long seconds = end_time.tv_sec - start_time.tv_sec;
    long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;

    if (nanoseconds < 0) {
        seconds--;
        nanoseconds += 1000000000;
    }

    double elapsed_seconds = (double)seconds + ((double)nanoseconds / 1000000000.0);
    
    if (i >= halfway) {
      total_time += elapsed_seconds;
    }
    
  }
  double avg_time = total_time / halfway;

  //2 floating point arrays - (2.0 * 4.0 * ((double)arrSize)
  double bandwidth = ((2.0 * 4.0 * ((double)arrSize)) / (1000000000.0)) / avg_time;

  //upper bound for dot product computation - (1 addition + 1 multiplication) * (size of floating point arrays) - 1
  //actual number of floating point operations for cblas_sdot function probably less due to advanced optimizations
  double throughput = (2 * ((double)arrSize) - 1) / avg_time;
  
  printf("N: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n", arrSize, avg_time, bandwidth, throughput);
  free(arr1);
  free(arr2);
  printf("Dot product result: %f\n", temp);
  return 0;
}

