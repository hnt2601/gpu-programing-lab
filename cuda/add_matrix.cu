#include <iostream>
#include <vector>
#include <stdio.h>


__global__ void MatAdd(float* A, float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("i %d, j %d\n", i, j);

    if (i < N && j < N) {
        int index = i * N + j;
        // printf("index %d\n", index);
        C[index] = A[index] + B[index];
    }
}

int main()
{
    int N = 16; // Size of matrix

    // Host matrices
    std::vector<float> h_A(N * N, 1);
    std::vector<float> h_B(N * N, 2);
    std::vector<float> h_C(N * N);

    std::vector<float> result(h_C.size());

    // Initialize matrices with sequential values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i + 1;
        h_B[i] = h_A[i] * 2;
    }

    // Device matrices
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel invocation with one block of N * N threads
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    std::cout << threadsPerBlock.x << std::endl;
    std::cout << threadsPerBlock.y << std::endl;

    std::cout << N / threadsPerBlock.x << std::endl;
    std::cout << N / threadsPerBlock.y << std::endl;

    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // Copy the result from device to host
    cudaMemcpy(result.data(), d_C, result.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display the result
    std::cout << "Matrix C after addition:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << result[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}