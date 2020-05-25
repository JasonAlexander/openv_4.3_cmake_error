//
// Created by developer on 5/24/20.
//

#include "saxpy.cuh"

__global__
void run(int n, float a, float *d_x, float *d_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_y[i] = a * d_x[i] + d_y[i];
}

void saxpy(int N, float a, float *x, float *y)
{
    float *d_x, *d_y;

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on N elements
    run<<<(N+255)/256, 256>>>(N, a, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}