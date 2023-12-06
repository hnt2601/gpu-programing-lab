#include <stdio.h>
#include <vector>


__global__ void VectorAdd(float* A, float* B, float* C, int n)
{
    int idx = threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];

        printf("idx = %d, A = %f, B = %f, C = %f\n", idx, A[idx], B[idx], C[idx]);
    }
}


int main()
{
    // Define two vectors
    std::vector<float> vectorA = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> vectorB = {5.0, 6.0, 7.0, 8.0};

    int size = static_cast<int>(vectorA.size());

    std::vector<float> result(size);

    printf("size = %d\n", size);
    
    // Kernel invocation with N threads
    int threadsPerBlock = size;

    // Declare device pointers
    float *A, *B, *C;

    // Allocate device memory
    cudaMalloc((void**)&A, size * sizeof(float));
    cudaMalloc((void**)&B, size * sizeof(float));
    cudaMalloc((void**)&C, size * sizeof(float));

     // Copy input vectors from host to device
    cudaMemcpy(A, vectorA.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, vectorB.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    VectorAdd<<<1, threadsPerBlock>>>(A, B, C, size);
    cudaDeviceSynchronize(); // Wait for the GPU to finish
    
    // Copy the result from device to host
    cudaMemcpy(result.data(), C, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        printf("i = %d, result = %f\n", i, result[i]);
    }

    // Free device memory
    cudaFree(C);
}