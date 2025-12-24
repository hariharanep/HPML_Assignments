#include <stdio.h>
#include <cudnn.h>

#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1

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

void initialize_I(double* I) {
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                I[c * (H * W) + (x) * W + (y)] = c * (x + y);
            }
        }
    }
}

int main() {
    size_t F_size = K * C * FH * FW * sizeof(double);
    size_t I_size = C * H * W * sizeof(double);
    size_t O_size = K * W * H * sizeof(double);

    double *h_F = (double*)malloc(F_size);
    double *h_I = (double*)malloc(I_size);
    double *h_O = (double*)malloc(O_size);

    initialize_F(h_F);
    initialize_I(h_I);

    double *d_F; 
    double *d_I; 
    double *d_O;
    cudaMalloc(&d_I, I_size);
    cudaMalloc(&d_F, F_size);
    cudaMalloc(&d_O, O_size);

    cudaMemcpy(d_I, h_I, I_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size, cudaMemcpyHostToDevice);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
		    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_DOUBLE,
                    1, C, H, W);
    
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
                    CUDNN_TENSOR_NCHW,
                    CUDNN_DATA_DOUBLE,
                    1, K, W, H);
    
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(filter_descriptor,
                    CUDNN_DATA_DOUBLE,
                    CUDNN_TENSOR_NCHW,
                    K, C, FH, FW);
    
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
		    P, P, 1, 1, 1, 1,
		    CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);
    
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    int requestedAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                           input_descriptor,
                                           filter_descriptor,
                                           convolution_descriptor,
                                           output_descriptor,
                                           requestedAlgoCount,
                                           &returnedAlgoCount,
                                           &perfResults);
    
    convolution_algorithm = perfResults.algo;

    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_descriptor,
                                            filter_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            convolution_algorithm,
                                            &workspace_bytes);
    
    // Allocate workspace
    void* d_workspace;
    if (workspace_bytes > 0) {
      cudaMalloc(&d_workspace, workspace_bytes);
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudnnConvolutionForward(cudnn,
                            &alpha,
                            input_descriptor,
                            d_I,
                            filter_descriptor,
                            d_F,
                            convolution_descriptor,
                            convolution_algorithm,
                            d_workspace,
                            workspace_bytes,
                            &beta,
                            output_descriptor,
                            d_O);
    
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
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (d_workspace) {
        cudaFree(d_workspace);
    }
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
    
    free(h_I);
    free(h_F);
    free(h_O);
    
    return 0;
}
