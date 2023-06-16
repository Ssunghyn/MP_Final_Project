#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "DS_definitions.h"
#include "DS_timer.h"

#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))
#define BLOCK_SIZE 1024
#define NUM_THREAD 8

__global__ void getRowSum(float* matrix, float* rowSum, int n, int rowOffset = 0)
{
	int row = blockIdx.x;
	int col = blockDim.x * blockIdx.y + threadIdx.x;
	if (col >= n || row + rowOffset >= n) return;
	__shared__ float subMatrixRow[BLOCK_SIZE];
	subMatrixRow[threadIdx.x] = matrix[ID2INDEX(row, col, n)];
	__syncthreads();
	int offset = BLOCK_SIZE / 2;
	while (offset > 0)
	{
		if (threadIdx.x < offset && col + offset < n)
			subMatrixRow[threadIdx.x] = __fadd_rn(subMatrixRow[threadIdx.x], subMatrixRow[threadIdx.x + offset]);
		offset /= 2;
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		atomicAdd(&rowSum[row], subMatrixRow[0]);
	}
}

float* generateLaplacianMatrix(float* affinityMatrix, int n) {
	float* result = new float[n * n];
	for (int i = 0; i < n; i++)
	{
		float sum = 0;
		for (int j = 0; j < n; j++)
		{
			if (i != j)
			{
				sum += affinityMatrix[ID2INDEX(i, j, n)];
				result[ID2INDEX(i, j, n)] = -affinityMatrix[ID2INDEX(i, j, n)];
			}
		}
		result[ID2INDEX(i, i, n)] = sum;
	}
	delete[] affinityMatrix;
	return result;
}

float* generateLaplacianMatrixOmp(float* affinityMatrix, int n) {
	float* result = new float[n * n];
	#pragma omp parallel num_threads(NUM_THREAD) 
	{
		#pragma omp for
		for (int i = 0; i < n; i++)
		{
			float sum = 0;
			for (int j = 0; j < n; j++)
			{
				if (i != j)
				{
					sum += affinityMatrix[ID2INDEX(i, j, n)];
					result[ID2INDEX(i, j, n)] = -affinityMatrix[ID2INDEX(i, j, n)];
				}
			}
			result[ID2INDEX(i, i, n)] = sum;
		}
	}
	delete[] affinityMatrix;
	return result;
}

float* generateLaplacianMatrixParallel(float* affinityMatrix, int n) {
	float* result = new float[n * n];
	float* rowSum = new float[n];
	float* dAffinityMatrix;
	float* dRowSum;
	dim3 grid(n, ceil(n / (float)BLOCK_SIZE), 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	cudaMalloc(&dRowSum, sizeof(float) * n);
	cudaMemset(dRowSum, 0, sizeof(float) * n);
	cudaMalloc(&dAffinityMatrix, sizeof(float) * n * n);
	cudaMemset(dAffinityMatrix, 0, sizeof(float) * n * n);
	cudaMemcpy(dAffinityMatrix, affinityMatrix, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	getRowSum << <grid, block >> > (dAffinityMatrix, dRowSum, n);
	cudaMemcpy(rowSum, dRowSum, sizeof(float) * n, cudaMemcpyDeviceToHost);
	#pragma omp parallel for num_threads(NUM_THREAD)
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i != j) {
				result[ID2INDEX(i, j, n)] = -affinityMatrix[ID2INDEX(i, j, n)];
			}
		}
	}
	cudaDeviceSynchronize();
	#pragma omp parallel for num_threads(NUM_THREAD)
	for (int i = 0; i < n; i++)
	{
		result[ID2INDEX(i, i, n)] = rowSum[i] - affinityMatrix[ID2INDEX(i, i, n)];
	}
	cudaFree(dRowSum);
	cudaFree(dAffinityMatrix);
	delete[] rowSum;
	delete[] affinityMatrix;
	return result;
}

float* generateLaplacianMatrixParallel2(float* affinityMatrix, int n) {
	float* result = new float[n * n];
	float* pinnedAffinityMatrix = new float[n * n];
	float* pinnedRowSum = new float[n];
	float* dAffinityMatrix;
	float* dRowSum;
	int streamNum = 32;
	int rowSize = (int)ceil(n / (float)streamNum);
	dim3 grid(ceil(n / (float)streamNum), ceil(n / (float)BLOCK_SIZE), 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	cudaMallocHost(&pinnedAffinityMatrix, sizeof(float) * n * n);
	cudaMallocHost(&pinnedRowSum, sizeof(float) * n);
	memset(pinnedAffinityMatrix, 0, sizeof(float) * n * n);
	memset(pinnedRowSum, 0, sizeof(float) * n);

	cudaMalloc(&dRowSum, sizeof(float) * n);
	cudaMalloc(&dAffinityMatrix, sizeof(float) * n * n);
	cudaMemset(dRowSum, 0, sizeof(float) * n);
	cudaMemset(dAffinityMatrix, 0, sizeof(float) * n * n);
		
	#pragma omp parallel for num_threads(NUM_THREAD)
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			pinnedAffinityMatrix[ID2INDEX(i, j, n)] = affinityMatrix[ID2INDEX(i, j, n)];
		}
	}

	cudaStream_t* stream = new cudaStream_t[streamNum];
	#pragma omp parallel num_threads(NUM_THREAD) 
	{
		#pragma omp for
		for(int i = 0; i < streamNum; i++) {
			cudaStreamCreate(&stream[i]);
			int affinityOffset = rowSize * n * i;
			int rowSumOffset = rowSize * i;
			cudaMemcpyAsync(dAffinityMatrix + affinityOffset, pinnedAffinityMatrix + affinityOffset, sizeof(float) * n * rowSize, cudaMemcpyHostToDevice, stream[i]);
			getRowSum << <grid, block, 0, stream[i] >> > (dAffinityMatrix + affinityOffset, dRowSum + rowSumOffset, n, rowSize * i);
			cudaMemcpyAsync(pinnedRowSum + rowSumOffset, dRowSum + rowSumOffset, sizeof(float) * rowSize, cudaMemcpyDeviceToHost, stream[i]);
		}
		#pragma omp for
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++) {
				if (i != j)
				{
					result[ID2INDEX(i, j, n)] = -pinnedAffinityMatrix[ID2INDEX(i, j, n)];
				}
			}
		}
	}
	cudaDeviceSynchronize();
	#pragma omp parallel for num_threads(NUM_THREAD)
	for (int i = 0; i < n; i++) {
		result[ID2INDEX(i, i, n)] = pinnedRowSum[i] - pinnedAffinityMatrix[ID2INDEX(i, i, n)];
	}

	#pragma omp parallel for num_threads(NUM_THREAD)
	for (int i = 0; i < streamNum; i++)
	{
		cudaStreamDestroy(stream[i]);
	}

	cudaFree(dRowSum);
	cudaFree(dAffinityMatrix);
	cudaFreeHost(pinnedAffinityMatrix);
	cudaFreeHost(pinnedRowSum);
	delete[] stream;
	delete[] affinityMatrix;
	return result;
}