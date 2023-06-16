#include "device_launch_parameters.h"
__global__ void getRowSum(float* matrix, float* rowSum, int n);
float* generateLaplacianMatrix(float* affinityMatrix, int n);
float* generateLaplacianMatrixParallel(float* affinityMatrix, int n);
float* generateLaplacianMatrixParallel2(float* affinityMatrix, int n);
float* generateLaplacianMatrixOmp(float* affinityMatrix, int n);