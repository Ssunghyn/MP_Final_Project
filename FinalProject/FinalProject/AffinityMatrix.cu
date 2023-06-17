

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include "DS_definitions.h"
#include <cmath>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <float.h>

#define SIGMA_DIMENSION 5

#define TEST_DATA_COUNT 10000
#define PRINT_RESULT false

#define GenDouble (rand() % 4 + ((float)(rand() % 100) / 100.0))

//OMP Properties
#define NUM_THREADS 8
#define OMP_OFFSET 16

//CUDA Properties
#define CUDA_THREADS_X 1024
#define CUDA_THREADS_Y 1
#define CUDA_THREADS_1D 1024
#define CUDA_THREADS_SELECTION_X 1024

#define CUDA_SHARED_MEMORY_CIR (CUDA_THREADS_X + CUDA_THREADS_X + 1)
#define CUDA_SHARED_MEMORY_CIR_DELTAS (CUDA_THREADS_X + 1)

const int NUM_BITS = 8;       // Number of bits to process per pass
const int NUM_DIGITS = 4;     // Number of digits (bytes) in each float value
const int NUM_BUCKETS = 256;  // Number of buckets (2^NUM_BITS)

__host__ inline float getDistance(float p1x, float p1y, float p2x, float p2y) {
    float x = pow(p2x - p1x, 2);
    float y = pow(p2y - p1y, 2);
    return sqrt(x + y);
}

__device__ inline float getDistanceDevice(float p1x, float p1y, float p2x, float p2y) {
    float x = __fmul_rn(p2x - p1x, p2x - p1x);
    float y = __fmul_rn(p2y - p1y, p2y - p1y);
    return sqrt(x + y);
}

float quickSelection(float* distances, float* pivot_left, float* pivot_right, int size, int k) {
    float pivot = distances[0];
    int same = 0;
    int pivot_left_count = 0;
    int pivot_right_count = 0;

    for (int i = 0; i < size; i++) {
        if (distances[i] < pivot) {
            pivot_left[pivot_left_count++] = distances[i];
        }
        else if (distances[i] > pivot) {
            pivot_right[pivot_right_count++] = distances[i];
        }
        else {
            same++;
        }
    }

    if (k <= pivot_left_count) {
        return quickSelection(pivot_left, pivot_left, pivot_right, pivot_left_count, k);
    }
    else if (k >= pivot_left_count + 1 && k <= pivot_left_count + same) {
        return pivot;
    }
    else {
        return quickSelection(pivot_right, pivot_left, pivot_right, pivot_right_count, k - pivot_left_count - same);
    }
}

void generateAffinityMatrix(float* point_x, float* point_y, const int point_count, float* result) {
    float* distance = new float[point_count * point_count];
    float* deltas = new float[point_count];


//#pragma omp parallel num_threads(NUM_THREADS)
    {
        float* pivot_left = new float[point_count];
        float* pivot_right = new float[point_count];

        // Get distance of 2 points
//#pragma omp for
        for (int p1 = 0; p1 < point_count; p1++) {
            for (int p2 = 0; p2 < point_count; p2++) {
                distance[p1 * point_count + p2] = getDistance(point_x[p1], point_y[p1], point_x[p2], point_y[p2]);
            }
        }

        //Pick [SIGMA_DIMENSION]th min point
//#pragma omp for
        for (int p1 = 0; p1 < point_count; p1++) {
            deltas[p1] = quickSelection(&distance[p1 * point_count], pivot_left, pivot_right, point_count, SIGMA_DIMENSION);
        }

        //Make affinity matrix
//#pragma omp for
        for (int p1 = 0; p1 < point_count; p1++) {
            for (int p2 = 0; p2 < point_count; p2++) {
                result[p1 * point_count + p2] = exp(-distance[p1 * point_count + p2] / (2 * deltas[p1] * deltas[p2]));
            }
        }

        delete[] pivot_left;
        delete[] pivot_right;
    }

#if PRINT_RESULT
    printf("Distance map(OpenMP)\n");
    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        for (int j = 0; j < TEST_DATA_COUNT; j++) {
            printf("%.4lf ", distance[i * TEST_DATA_COUNT + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#if PRINT_RESULT
    printf("Deltas map(OpenMP)\n");
    for (int j = 0; j < TEST_DATA_COUNT; j++) {
        printf("%.4lf ", deltas[j]);
    }
    printf("\n");
    printf("\n");
#endif

    //Free memory
    delete[] deltas;
}

__global__ void generateDistanceMatrix_v1(float* point_x, float* point_y, const int point_count, float* distance) {
    unsigned int p2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (p1 >= point_count || p2 >= point_count) return;

    distance[p1 * point_count + p2] = getDistanceDevice(point_x[p1], point_y[p1], point_x[p2], point_y[p2]);
/*
    __shared__ float sP[CUDA_SHARED_MEMORY_CIR * CUDA_THREADS_Y];

    float px = point_x[p2];
    float py = point_y[p2];

    sP[CUDA_SHARED_MEMORY_CIR * threadIdx.y + threadIdx.x] = px;
    sP[CUDA_SHARED_MEMORY_CIR * threadIdx.y + CUDA_THREADS_X + threadIdx.x] = py;

    __syncthreads();

    distance[p1 * point_count + p2] = getDistanceDevice(sP[threadIdx.y + threadIdx.y * CUDA_SHARED_MEMORY_CIR], sP[threadIdx.y + CUDA_THREADS_X + threadIdx.y * CUDA_SHARED_MEMORY_CIR], sP[threadIdx.x + threadIdx.y * CUDA_SHARED_MEMORY_CIR], sP[threadIdx.x + CUDA_THREADS_X + threadIdx.y * CUDA_SHARED_MEMORY_CIR]);
    */
}

__global__ void generateAffinityMatrix_v1(float* distance, float* deltas, const int point_count, float* result) {
    unsigned int p2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (p1 >= point_count || p2 >= point_count) return;

    /*__shared__ float sD[CUDA_SHARED_MEMORY_CIR * CUDA_THREADS_Y];

    sD[CUDA_SHARED_MEMORY_CIR * threadIdx.y + threadIdx.x] = deltas[p2];

    __syncthreads();

    float deltasMul = __fmul_rn(__fmul_rn(sD[threadIdx.y + CUDA_SHARED_MEMORY_CIR * threadIdx.y], sD[threadIdx.x + CUDA_SHARED_MEMORY_CIR * threadIdx.y]), 2);

    result[p1 * point_count + p2] = exp(__fdiv_rn(-distance[p1 * point_count + p2], deltasMul));*/

    float deltasMul = __fmul_rn(__fmul_rn(deltas[p1], deltas[p2]), 2);
    result[p1 * point_count + p2] = exp(__fdiv_rn(-distance[p1 * point_count + p2], deltasMul));
}

void generateAffinityMatrix_cuda(float* point_x, float* point_y, const int point_count, float* result, float* d_result) {
    float* distance, * d_distance, * d_sorted_distance;
    float* d_point_x, * d_point_y;
    float* d_deltas;

    float* pivot_left = new float[point_count];
    float* pivot_right = new float[point_count];
    size_t distance_mem_size = sizeof(float) * point_count * point_count;
    size_t point_mem_size = sizeof(float) * point_count;
    cudaMalloc(&d_distance, distance_mem_size);
    cudaMalloc(&d_point_x, point_mem_size);
    cudaMalloc(&d_point_y, point_mem_size);

    cudaMemcpy(d_point_x, point_x, point_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_point_y, point_y, point_mem_size, cudaMemcpyHostToDevice);

    dim3 dimGridDistance(ceil((float)point_count / CUDA_THREADS_X), ceil((float)point_count / CUDA_THREADS_Y));
    dim3 dimBlockDiatance(CUDA_THREADS_X, CUDA_THREADS_Y);
    generateDistanceMatrix_v1 << <dimGridDistance, dimBlockDiatance >> > (d_point_x, d_point_y, point_count, d_distance);
    distance = (float*)malloc(distance_mem_size);
    float* deltas = new float[point_count];
    cudaMalloc(&d_result, distance_mem_size);
    cudaMalloc(&d_deltas, point_mem_size);
    cudaDeviceSynchronize();

    cudaStream_t* streams = new cudaStream_t[point_count];

#if PRINT_RESULT
    printf("Distance map(CUDA)\n");
    cudaMemcpy(distance, d_distance, distance_mem_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        for (int j = 0; j < TEST_DATA_COUNT; j++) {
            printf("%.4lf ", distance[i * TEST_DATA_COUNT + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#pragma omp parallel num_threads(NUM_THREADS) 
    {
#pragma omp for
        for (int p1 = 0; p1 < point_count; p1++) {
            //cudaStreamCreate(&streams[p1]);
            //cudaMemcpyAsync(&distance[p1 * point_count], &d_distance[p1 * point_count], point_mem_size, cudaMemcpyDeviceToHost, streams[p1]);
            //cudaStreamSynchronize(streams[p1]);
            cudaMemcpy(&distance[p1 * point_count], &d_distance[p1 * point_count], point_mem_size, cudaMemcpyDeviceToHost);
            deltas[p1] = quickSelection(&distance[p1 * point_count], pivot_left, pivot_right, point_count, SIGMA_DIMENSION);
        }
    }

#if PRINT_RESULT
    printf("Deltas map(CUDA)\n");
    for (int j = 0; j < TEST_DATA_COUNT; j++) {
        printf("%.4lf ", deltas[j]);
    }
    printf("\n\n");
#endif

    cudaMemcpy(d_deltas, deltas, point_mem_size, cudaMemcpyHostToDevice);

    //Make affinity matrix
    generateAffinityMatrix_v1 << <dimGridDistance, dimBlockDiatance >> > (d_distance, d_deltas, point_count, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_result, distance_mem_size, cudaMemcpyDeviceToHost);
}

int affinMain()
//int main()
{
    cudaFree(0);
    DS_timer timer(2);
    timer.setTimerName(0, (char*)"Serial");
    timer.setTimerName(1, (char*)"OpenMP + CUDA");

    float point_x[TEST_DATA_COUNT];
    float point_y[TEST_DATA_COUNT];
    float* result = new float[TEST_DATA_COUNT * TEST_DATA_COUNT];
    float* result_cuda = new float[TEST_DATA_COUNT * TEST_DATA_COUNT];
    float* d_result = NULL;


#if PRINT_RESULT
    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        point_x[i] = GenDouble;
        point_y[i] = GenDouble;
        printf("(%lf, %lf)\n", point_x[i], point_y[i]);
    }
#endif

    timer.onTimer(0);
    generateAffinityMatrix(point_x, point_y, TEST_DATA_COUNT, result);
    timer.offTimer(0);

    timer.onTimer(1);
    generateAffinityMatrix_cuda(point_x, point_y, TEST_DATA_COUNT, result_cuda, d_result);
    timer.offTimer(1);

#if PRINT_RESULT
    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        for (int j = 0; j < TEST_DATA_COUNT; j++) {
            printf("%.4lf ", result[i * TEST_DATA_COUNT + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        for (int j = 0; j < TEST_DATA_COUNT; j++) {
            printf("%.4lf ", result_cuda[i * TEST_DATA_COUNT + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        for (int j = 0; j < TEST_DATA_COUNT; j++) {
            if (result_cuda[i * TEST_DATA_COUNT + j] - result[i * TEST_DATA_COUNT + j] > 0.0001) {
                printf("(%d, %d), %.4lf, %.4lf \n", j, i, result[i * TEST_DATA_COUNT + j], result_cuda[i * TEST_DATA_COUNT + j]);
            }
        }
    }
    printf("\n");
#endif

    bool same = true;

    for (int i = 0; i < TEST_DATA_COUNT; i++) {
        for (int j = 0; j < TEST_DATA_COUNT; j++) {
            if (result_cuda[i * TEST_DATA_COUNT + j] - result[i * TEST_DATA_COUNT + j] > 0.0001) {
                same = false;
            }
            if (!same) continue;
        }
        if (!same) continue;
    }

    if(!same) printf("Result is not same \n");
    else printf("Result is same \n");

    timer.printTimer();

    return 0;
}