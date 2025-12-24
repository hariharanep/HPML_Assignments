/// Multiplies two matrices using CUDA: A x B = C
///

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
// Each thread computes a 2x2 block of C
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // The base row and column for this thread's 2x2 block of C
  int row = 2 * thread_row;
  int col = 2 * thread_col;

  // thread (thread_row, thread_col) computes elements:
  // C[2 * thread_row,     2 * thread_col]     C[2 * thread_row,     2 * thread_col + 1]
  // C[2 * thread_row + 1, 2 * thread_col]     C[2 * thread_row + 1, 2 * thread_col + 1]

  // Each bloock computes one FOOTPRINT_SIZE x FOOTPRINT_SIZE sub matrix of C = Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes 4 values of Csub unrolled into separate variables
  float Cvalue00 = 0;
  float Cvalue01 = 0;
  float Cvalue10 = 0;
  float Cvalue11 = 0;

  // Loop over all sub matrices required to compute Csub
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Shared memory - now FOOTPRINT_SIZE x FOOTPRINT_SIZE
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    
    
    int thread_id = thread_row * BLOCK_SIZE + thread_col;
    // Each thread loads 4 indices
    // Thread 0 loads indices 0,256,512,768
    // Thread 1 loads indices 1,257,513,769
    // etc. This ensures consecutive threads load consecutive memory in each iteration
    #pragma unroll
    for(int i = 0; i < 4; i++) {
      int idx = thread_id + i * (BLOCK_SIZE * BLOCK_SIZE);
      int new_row = idx / FOOTPRINT_SIZE;
      int new_col = idx % FOOTPRINT_SIZE;
      shared_A[new_row][new_col] = Asub[new_row * A.stride + new_col];
      shared_B[new_row][new_col] = Bsub[new_row * B.stride + new_col];
    }

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Compute the 2x2 block of Csub values in unrolled format
#pragma unroll
    for(int i = 0; i < FOOTPRINT_SIZE; i++) {
      float a0 = shared_A[row][i];
      float a1 = shared_A[row + 1][i];
      float b0 = shared_B[i][col];
      float b1 = shared_B[i][col + 1];

      Cvalue00 += a0 * b0;
      Cvalue01 += a0 * b1;
      Cvalue10 += a1 * b0;
      Cvalue11 += a1 * b1;
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write the 2x2 block to GLOBAL memory in a coalesced access pattern
  Csub[row * C.stride + col] = Cvalue00;
  Csub[row * C.stride + col + 1] = Cvalue01;
  Csub[(row + 1) * C.stride + col] = Cvalue10;
  Csub[(row + 1) * C.stride + (col + 1)] = Cvalue11;
}

