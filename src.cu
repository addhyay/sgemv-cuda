#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define MAXBLOCKSIZE 1024

void displayMat(int *A, int M, int N){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            printf("%d\t", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void displayVec(int *A, int N){
    for(int i = 0; i < N; i++){
        printf("%d\t", A[i]);
    }
    printf("\n");
}

__global__ void sgemv_v1(int  *A, int *x, int *y, int M, int N){
    unsigned row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (unsigned col = 0; col < N; col++) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

int main() {
    unsigned m = 5, n = 7;
    int *cpu_a, *cpu_x, *cpu_y;
    int *gpu_a, *gpu_x, *gpu_y;

    // allocate memory to cpu variables
    cpu_a = (int*) malloc(sizeof(int)*m*n);
    cpu_x = (int*) malloc(sizeof(int)*n);
    cpu_y = (int*) malloc(sizeof(int)*n);

    // allocate memory to gpu variables
    cudaMalloc(&gpu_a, sizeof(int)*m*n);
    cudaMalloc(&gpu_x, sizeof(int)*n);
    cudaMalloc(&gpu_y, sizeof(int)*n);

    // populate cpu_a
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            // cpu_a[i * n + j] = rand() % (m * n + 1);
            cpu_a[i * n + j] = i + j;
        }
    }

    // populate cpu_x
    for(int i = 0; i < n; i++){
        // cpu_x[i] = rand() % (m * n + 1);
        cpu_x[i] = i;
    }

    displayMat(cpu_a, m, n);
    displayVec(cpu_x, n);

    // copy data to gpu
    cudaMemcpy(gpu_a, cpu_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_x, cpu_x, sizeof(int)*n, cudaMemcpyHostToDevice);

    // call kernel
    unsigned nBlocks = ceil((float)(m * n + MAXBLOCKSIZE - 1) / MAXBLOCKSIZE);
    sgemv_v1<<<nBlocks, MAXBLOCKSIZE>>>(gpu_a, gpu_x, gpu_y, m, n);

    // copy result
    cudaMemcpy(cpu_y, gpu_y, sizeof(int)*n, cudaMemcpyDeviceToHost);

    // print result
    displayVec(cpu_y, n);
    printf("\n");

    return 0;
}
