#include <stdio.h>


__global__ void ScalarAdd(float A, float B, float* C)
{
    *C = A + B;
}


int main()
{
    // Kernel invocation with N threads
    int threadsPerBlock = 1;

    // Declare device pointers
    float A, B, *C;
    float result;

    A = 1.5;
    B = 2.5;

    // Allocate device memory
    cudaMalloc((void**)&C, sizeof(float));

    ScalarAdd<<<1, threadsPerBlock>>>(A, B, C);
    cudaDeviceSynchronize(); // Wait for the GPU to finish
    
    // Copy the result from device to host
    cudaMemcpy(&result, C, sizeof(float), cudaMemcpyDeviceToHost);

    printf("result = %f\n", result);

    // Free device memory
    cudaFree(C);
}