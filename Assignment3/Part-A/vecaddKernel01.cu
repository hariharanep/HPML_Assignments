///
/// vecAddKernel01.cu
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
///

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int totalThreads = gridDim.x * blockDim.x;
    int totalSize = totalThreads * N;
    int threadStartIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // This loop ensures consecutive threads access consecutive memory in each iteration.
    // thread 0 accesses index 0 of A,B,C
    // thread 1 accesses index 1 of A,B,C
    // thread 2 accesses index 2 of A,B,C
    // ...
    // last thread accesses last index of A,B,C
    int i;
    for( i=threadStartIndex; i<totalSize; i+=totalThreads ){
        C[i] = A[i] + B[i];
    }
}

