#include <stdio.h>


__global__ void kernel()
{
    printf("CUDA thread %d\n", threadIdx.x);
}

int main()
{
    int threadsPerBlock = 1;
    kernel<<<1, threadsPerBlock>>>();
    cudaDeviceSynchronize(); // Wait for the GPU to finish
}