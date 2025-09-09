#include <cstddef>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// #include "include/coalesced_warp_2.cuh"
// #include "include/coalesced_warpblock_3.cuh"
#include "include/cublas_0.cuh"
// #include "include/naive_1.cuh"
#include "include/utils.cuh"
#include "include/vectorized_4.cuh"

/*
 * Benchmarks a kernel against cuBLAS for different sizes
 */

void benchmark_kernel_for_sizes(int minM, int maxM, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
    FILE *gflops_file = fopen("benchmarks/kernel_4_vs_cublas-gflops.txt", "w");
    FILE *memory_file = fopen("benchmarks/kernel_4_vs_cublas-memory.txt", "w");

    // error(s) while opening the file
    if (gflops_file == NULL) {
        perror("[Error] opening the file for GFLOPS.\n");
    }
    if (memory_file == NULL) {
        perror("[Error] opening the file for memory bandwidth.\n");
    }

    for (int M = minM; M <= maxM; M *= 2) {
        int N = 2 * M;      // matrix size : (M, N)

        printf("--------------- Running benchmark for M = %d ---------------\n", M);

        size_t matsize = M * N;
        size_t vecsize = N;
        size_t mat_totalsize = matsize * sizeof(float);
        size_t vec_totalsize = vecsize * sizeof(float);

        // allocate host
        float *mat = (float *) malloc(mat_totalsize);
        float *vec = (float *) malloc(vec_totalsize);
        float *res = (float *) malloc(sizeof(float) * M);

        for (size_t i = 0; i < matsize; i++) {
            mat[i] = random_normal_clamped(-10.f, 10.f);
            if (i < vecsize) {
                vec[i] = random_normal_clamped(-10.f, 10.f);
            }
        }

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        float ms = 0.0f;

        // allocate device
        float *matd, *vecd, *resd;
        cudaEventRecord(start);             // start the timer
        CUDA_CHECK(cudaMalloc((void **)&matd, mat_totalsize));
        CUDA_CHECK(cudaMalloc((void **)&vecd, vec_totalsize));
        CUDA_CHECK(cudaMalloc((void **)&resd, sizeof(float) * M));
        cudaEventRecord(end);               // end the timer
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        printf(">> GPU allocation time: %f ms\n", ms);

        // copy host to device
        cudaEventRecord(start);             // start the timer
        CUDA_CHECK(cudaMemcpy(matd, mat, mat_totalsize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vecd, vec, vec_totalsize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(resd, res, sizeof(float) * M, cudaMemcpyHostToDevice));
        cudaEventRecord(end);               // end the timer
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        printf(">> Host to device transfer time: %f ms\n", ms);

        // run cuBLAS kernel and write results to file
        float mscub = run_kernel_cublas_sgemv(matd, vecd, resd, M, N, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
        float gflopscub = compute_gflops(M, N, mscub);
        float mem_bandcub = compute_peak_memory_bandwidth(M, N, mscub, THEORETICAL_MAX_MEMORY_BANDWIDTH);

        // run custom kernel and write result to file
        ms = run_kernel_vectorized_sgmev(matd, vecd, resd, M, N, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
        float gflops = compute_gflops(M, N, ms);
        float mem_band = compute_peak_memory_bandwidth(M, N, ms, THEORETICAL_MAX_MEMORY_BANDWIDTH);

        fprintf(gflops_file, "%d %f %f\n", M, gflops, gflopscub);
        fprintf(memory_file, "%d %f %f\n", M, mem_band, mem_bandcub);

        // copy device to host
        cudaEventRecord(start);             // start the timer
        CUDA_CHECK(cudaMemcpy(res, resd, sizeof(float) * M, cudaMemcpyDeviceToHost));
        cudaEventRecord(end);               // end the timer
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        printf(">> Device to host transfer time: %f ms\n", ms);

        // cleanup
        cudaFree(matd);
        cudaFree(vecd);
        cudaFree(resd);
        free(mat);
        free(vec);
        free(res);
    }

    fclose(gflops_file);
    fclose(memory_file);
 }

int main() {
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, 0);
     int cudaCores = prop.multiProcessorCount * 128;
     // float clockGHZ = prop.clockRate / 1e6;

     int clockKHZ = 0;
     int memClockKHZ = 0;
     int memBusWidth = 0;

     // query
     cudaDeviceGetAttribute(&clockKHZ, cudaDevAttrClockRate, 0);
     cudaDeviceGetAttribute(&memClockKHZ, cudaDevAttrMemoryClockRate, 0);
     cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);

     float clockGHZ = clockKHZ / 1e6;

     float THEORETICAL_MAX_GFLOPS = 2 * cudaCores * clockGHZ;
     float THEORETICAL_MAX_MEMORY_BANDWIDTH = (2.0f * memClockKHZ * 1000 * (memBusWidth / 8.0f)) / 1e9f;

     benchmark_kernel_for_sizes(4096, 4096, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
 }
