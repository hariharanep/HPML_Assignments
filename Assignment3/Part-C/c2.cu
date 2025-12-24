#include <stdio.h>

#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1

#define TILE_SIZE 16

__global__ void convolution_kernel(double* O, double* IO, double* F) {
    __shared__ double s_tile[C][TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    for (int c = 0; c < C; c++) {
        s_tile[c][threadIdx.x][threadIdx.y] = IO[c * (H + 2*P) * (W + 2*P) + x * (W + 2*P) + y];
    }

    __syncthreads();
        
    double sum = 0.0;

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < FW; i++) {
            for (int j = 0; j < FH; j++) {
                int filter_idx = k * (C*FH*FW) + c * (FH*FW) + (FW-1-i) * FW + (FH-1-j);
                
                // Check if we can use shared memory
                int local_x = threadIdx.x + i;
                int local_y = threadIdx.y + j;
                
                double input_val;
                if (local_x < TILE_SIZE && local_y < TILE_SIZE) {
                    // Use shared memory
                    input_val = s_tile[c][local_x][local_y];
                } else {
                    int input_idx = c * ((H + 2*P) * (W + 2*P)) + (x + i) * (W + 2*P) + (y + j);
                    input_val = IO[input_idx];
                }
                
                sum += F[filter_idx] * input_val;
            }
        }
    }
    
    int output_idx = k * (W * H) + x * W + y;
    O[output_idx] = sum;
}

void initialize_F(double* F) {
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    F[k * (C * FH * FW) + c * (FH * FW) + i * (FW) + j] = (c + k) * (i + j);
                }
            }
        }
    }
}

void initialize_IO(double* IO) {
    int height = H + 2*P;
    int width = W + 2*P;
    for (int i = 0; i < C * height * width; i++) {
        IO[i] = 0.0;
    }


    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                IO[c * (height * width) + (x + P) * width + (y + P)] = c * (x + y);
            }
        }
    }
}

int main() {
    size_t F_size = K * C * FH * FW * sizeof(double);
    size_t IO_size = C * (H + 2*P) * (W + 2*P) * sizeof(double);
    size_t O_size = K * W * H * sizeof(double);

    double *h_F = (double*)malloc(F_size);
    double *h_IO = (double*)malloc(IO_size);
    double *h_O = (double*)malloc(O_size);

    initialize_F(h_F);
    initialize_IO(h_IO);


    double *d_F; 
    double *d_IO; 
    double *d_O;
    cudaMalloc(&d_IO, IO_size);
    cudaMalloc(&d_F, F_size);
    cudaMalloc(&d_O, O_size);

    cudaMemcpy(d_IO, h_IO, IO_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size, cudaMemcpyHostToDevice);

    //Configuration that allows each thread to be responsible for exactly one output element
    dim3 blockDim(16, 16, 1); // 256 threads per block
    dim3 gridDim(64, 64, 64); // 262144 blocks per grid

    //Warm Up
    convolution_kernel<<<gridDim, blockDim>>>(d_O, d_IO, d_F);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution_kernel<<<gridDim, blockDim>>>(d_O, d_IO, d_F);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_O, d_O, O_size, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (int i = 0; i < K * H * W; i++) {
        checksum += h_O[i];
    }
    printf("Checksum: %.6f\n", checksum);
    printf("Kernel Execution Time: %.6f ms\n", milliseconds);

    free(h_IO);
    free(h_F);
    free(h_O);
    cudaFree(d_IO);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
