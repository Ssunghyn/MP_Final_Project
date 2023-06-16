#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "omp.h"

#include "DS_timer.h"

#define BLOCK_SIZE_1 (512)
#define BLOCK_SIZE_2 (32)
#define GEN_FLOAT (rand() % 10 + (rand() % 10) / 10.0)
#define NUM_THREADS (8)

#define M 10000
#define K 10000
#define N 10000

#define tol 1e-5

__global__ void matMul_Kernel(float* _A, float* _B, float* _C)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row * N + col;
    float val = 0;

    __shared__ float sA[BLOCK_SIZE_2][BLOCK_SIZE_2];
    __shared__ float sB[BLOCK_SIZE_2][BLOCK_SIZE_2];

    for (int i = 0; i < ceil((float)K / BLOCK_SIZE_2); i++)
    {
        int stride = BLOCK_SIZE_2 * i;

        if (row >= M || stride + threadIdx.x >= K)
            sA[threadIdx.y][threadIdx.x] = 0;
        else
            sA[threadIdx.y][threadIdx.x] = _A[row * K + stride + threadIdx.x];
        if (col >= N || stride + threadIdx.y >= K)
            sB[threadIdx.y][threadIdx.x] = 0;
        else
            sB[threadIdx.y][threadIdx.x] = _B[(stride + threadIdx.y) * N + col];

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE_2; j++)
        {
            val += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        __syncthreads();  // 계산이 완료될 때까지 대기
    }

    if (row >= M || col >= N) return;  // 범위 이외의 부분 예외처리

    _C[index] = val;
}

__global__ void norm_Kernel(float* _v, float* ans, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float localVal[256];
    if (tid < n)
    {
        localVal[threadIdx.x] = _v[tid] * _v[tid];
        __syncthreads();

        int offset = 256 / 2;
        while (offset > 0)
        {
            if (threadIdx.x < offset)
            {
                localVal[threadIdx.x] += localVal[threadIdx.x + offset];
            }
            offset /= 2;
            __syncthreads();
        }

        if (threadIdx.x == 0)
            atomicAdd(ans, localVal[0]);
    }
}

void matMul(float* A, float* B, float* output, int n)
{
    float* tmp = new float[n * n];
    for (int i = 0; i < n * n; i++)
        tmp[i] = 0.0;

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                tmp[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    for (int i = 0; i < n * n; i++)
        output[i] = tmp[i];

    delete tmp;
}

float norm(float* v, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += *(v + i) * *(v + i);
    }
    return sqrt(result);
}

float normMulti(float* v, int n)
{
    double result = 0.0;
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:result)
    for (int i = 0; i < n; i++) {
        result += *(v + i) * *(v + i);
    }
    return sqrt(result);
}

float dot(float* A, float* B, int n)
{
    float result = 0.0;
    for (int i = 0; i < n; i++)
        result += *(A + i) * *(B + i);
    return result;
}

float dotMulti(float* A, float* B, int n)
{
    float result = 0.0;
#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:result)
    for (int i = 0; i < n; i++)
        result += *(A + i) * *(B + i);
    return result;
}

void QR_Decomposition_Multi(float* A, float* Q, float* R, int n)
{
    float* uT = new float[n * n];
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; i++)
        uT[i] = A[i];

    float uTNorm = normMulti(uT, n);
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        Q[i] = uT[i] / uTNorm;
    }

    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            uT[i * n + j] = A[i * n + j];

        for (int j = 0; j < i; j++)
        {
            float dotAQ = dotMulti(A + i * n, Q + j * n, n);
            for (int k = 0; k < n; k++)
            {
                uT[i * n + k] -= dotAQ * Q[j * n + k];
            }
        }

        float norm_u_i = normMulti(uT + i * n, n);
        for (int j = 0; j < n; j++)
        {
            Q[i * n + j] = uT[i * n + j] / norm_u_i;
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            R[i * n + j] = dotMulti(A + j * n, Q + i * n, n);
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            float tmp = Q[i * n + j];
            Q[i * n + j] = Q[j * n + i];
            Q[j * n + i] = tmp;
        }
    }

    delete uT;
}

void QR_Decomposition(float* A, float* Q, float* R, int n) {
    float* uT = new float[n * n];

    for (int i = 0; i < n; i++)
        uT[i] = A[i];

    for (int i = 0; i < n; i++)
    {
        Q[i] = uT[i] / norm(uT, n);
    }

    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            uT[i * n + j] = A[i * n + j];

        for (int j = 0; j < i; j++)
        {
            float dotAQ = dot(A + i * n, Q + j * n, n);
            for (int k = 0; k < n; k++)
            {
                uT[i * n + k] -= dotAQ * Q[j * n + k];
            }
        }

        float norm_u_i = norm(uT + i * n, n);

        for (int j = 0; j < n; j++)
        {
            Q[i * n + j] = uT[i * n + j] / norm_u_i;
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            R[i * n + j] = dot(A + j * n, Q + i * n, n);
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            float tmp = Q[i * n + j];
            Q[i * n + j] = Q[j * n + i];
            Q[j * n + i] = tmp;
        }
    }

    delete uT;
}

void normTestMain()
//int main()
{
    DS_timer timer(5);
    timer.setTimerName(0, "CPU Parallel");
    timer.setTimerName(1, "GPU Kernel");
    timer.setTimerName(2, "CPU Reduction");
    timer.setTimerName(3, "CPU Dot Serial");
    timer.setTimerName(4, "CPU Dot Parallel");

    float* vec = new float[N];
    float* vec2 = new float[N];
    float* d_vec;   float* d_ans; float gpuAns;

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++)
    {
        vec[i] = GEN_FLOAT;
        vec2[i] = GEN_FLOAT;
    }

    timer.onTimer(0);
    float cpuAns = norm(vec, N);
    timer.offTimer(0);

    cudaMalloc(&d_vec, sizeof(float) * N);  cudaMemset(d_vec, 0, sizeof(float) * N);
    cudaMalloc(&d_ans, sizeof(float));  cudaMemset(d_ans, 0, sizeof(float));

    timer.onTimer(1);
    cudaMemcpy(d_vec, vec, sizeof(float) * N, cudaMemcpyHostToDevice);
    norm_Kernel << <ceil(N / 256.0), 256 >> > (d_vec, d_ans, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpuAns, d_ans, sizeof(float), cudaMemcpyDeviceToHost);
    gpuAns = sqrt(gpuAns);
    timer.offTimer(1);

    timer.onTimer(2);
    float cpuReduction = normMulti(vec, N);
    timer.offTimer(2);

    timer.onTimer(3);
    float cpuDot = dot(vec, vec2, N);
    timer.offTimer(3);

    timer.onTimer(4);
    float cpuParallelDot = dotMulti(vec, vec2, N);
    timer.offTimer(4);
    timer.printTimer();

    if (cpuAns == gpuAns && cpuAns == cpuReduction)
        printf("Norm Answer is Correct! %f\n", gpuAns);
    else
        printf("Norm Parallel is not Correct! %f, %f, %f\n", cpuAns, gpuAns, cpuReduction);

    if (cpuDot == cpuParallelDot)
        printf("Dot Answer is Correct! %f\n", cpuDot);
    else
        printf("Dot Parallel is not Correct! %f, %f\n", cpuDot, cpuParallelDot);

    cudaFree(d_vec);    cudaFree(d_ans);
    delete vec;     delete vec2;
}

void findEigenQR(float* A, float* eigValue, float* eigVector, int n, int maxIter = 300)
{
    float* A_old = new float[n * n];
    float* Q = new float[n * n];
    float* R = new float[n * n];

    for (int i = 0; i < n; i++)
    {
        eigVector[i * n + i] = 1.0;
    }

    for (int i = 0; i < n * n; i++)
        A_old[i] = A[i];

    int count = 0;
    while (count < maxIter)
    {
        QR_Decomposition(A_old, Q, R, n);
        matMul(eigVector, Q, eigVector, n);
        matMul(R, Q, A_old, n);
        count++;
    }

    for (int i = 0; i < n; i++) {
        eigValue[i] = A_old[i * n + i];
    }
}

void findEigenQR_Multi(float* A, float* eigValue, float* eigVector, int n, int maxIter = 300)
{
    float* A_old = new float[n * n];
    float* Q = new float[n * n];
    float* R = new float[n * n];
    /*
    float* d_Q; float* d_R; float* d_oldA;  float* d_eigVec;    float* tmp;
    cudaMalloc(&d_oldA, sizeof(float) * n * n);
    cudaMalloc(&d_Q, sizeof(float) * n * n);
    cudaMalloc(&d_R, sizeof(float) * n * n);
    cudaMalloc(&d_eigVec, sizeof(float) * n * n);
    cudaMalloc(&tmp, sizeof(float) * n * n);
    */
    /*
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        eigVector[i * n + i] = 1.0;
    }
    */
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n * n; i++)
    {
        if (i % n == i / n)
        {
            eigVector[i] = 1.0;
        }
        A_old[i] = A[i];
    }


    //dim3 block(BLOCK_SIZE_V2, BLOCK_SIZE_V2);
    //dim3 grid(ceil((float)n / BLOCK_SIZE_V2), ceil((float)n / BLOCK_SIZE_V2));

    int count = 0;
    while (count < maxIter)
    {
        QR_Decomposition_Multi(A_old, Q, R, n);
        matMul(eigVector, Q, eigVector, n);
        matMul(R, Q, A_old, n);
        /*
        cudaMemcpy(d_oldA, A_old, sizeof(float) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_eigVec, eigVector, sizeof(float) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, Q, sizeof(float) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_R, R, sizeof(float) * n * n, cudaMemcpyHostToDevice);

        matMul_Kernel << <grid, block>> > (d_eigVec, d_Q, tmp);
        matMul_Kernel << <grid, block >> > (d_R, d_Q, d_oldA);

        cudaMemcpy(A_old, d_oldA, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(eigVector, tmp, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(Q, d_Q, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(R, d_R, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
        */
        count++;
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; i++) {
        eigValue[i] = A_old[i * n + i];
    }

    //cudaFree(d_oldA);   cudaFree(d_Q);  cudaFree(d_R);  cudaFree(d_eigVec); cudaFree(tmp);
    delete A_old, Q, R;
}

__global__ void normalize(float* uT, float* Q, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float localVal[BLOCK_SIZE_1];
    float norm = 0.0;

    int id = threadIdx.x;
    localVal[id] = 0.0;

    while (id < n)
    {
        localVal[threadIdx.x] += uT[id] * uT[id];
        id += BLOCK_SIZE_1;
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    while (offset > 0)
    {
        if (threadIdx.x < offset)
            localVal[threadIdx.x] += localVal[threadIdx.x + offset];
        offset /= 2;
        __syncthreads();
    }

    if (idx < n) {
        norm = sqrt(localVal[0]);
        Q[idx] = uT[idx] / norm;
    }
}

__global__ void computeDotProduct(float* A, float* Q, float* uT, int n, int i, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float localVal[BLOCK_SIZE_1];
    float dotAQ = 0.0;

    int id = threadIdx.x;
    localVal[id] = 0;
    while (id < n)
    {
        localVal[threadIdx.x] += A[i * n + id] * Q[j * n + id];
        id += BLOCK_SIZE_1;
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    while (offset > 0)
    {
        if (threadIdx.x < offset)
            localVal[threadIdx.x] += localVal[threadIdx.x + offset];
        offset /= 2;
        __syncthreads();
    }

    dotAQ = localVal[0];

    if (idx < n) {

        uT[i * n + idx] -= dotAQ * Q[j * n + idx];
    }
}

__global__ void normalizeU(float* uT, float* Q, int n, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float localVal[BLOCK_SIZE_1];
    float norm = 0.0;

    int id = threadIdx.x;
    localVal[id] = 0.0;
    while (id < n)
    {
        localVal[threadIdx.x] += uT[i * n + id] * uT[i * n + id];
        id += BLOCK_SIZE_1;
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    while (offset > 0)
    {
        if (threadIdx.x < offset)
            localVal[threadIdx.x] += localVal[threadIdx.x + offset];
        offset /= 2;
        __syncthreads();
    }

    if (idx < n) {
        norm = sqrt(localVal[0]);
        Q[i * n + idx] = uT[i * n + idx] / norm;
    }
}

__global__ void computeDotProductR(float* A, float* Q, float* R, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float localVal[BLOCK_SIZE_2];
    localVal[i] = 0.0;
    if (i < n && j < n)
    {
        localVal[i] += A[j * n + k] * Q[i * n + k]
    }

    if (i < n && j >= i && j < n)
    {
        float dotAQ = 0.0f;

        /*
        for (int k = 0; k < n; k++)
        {
            dotAQ += A[j * n + k] * Q[i * n + k];
        }
        */
        R[i * n + j] = dotAQ;
    }
}


__global__ void transpose(float* Q, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n && idx > idy) {
        float tmp = Q[idx * n + idy];
        Q[idx * n + idy] = Q[idy * n + idx];
        Q[idy * n + idx] = tmp;
    }
}

void QR_Decomposition_CUDA(float* A, float* Q, float* R, int n) {
    float* d_A, * d_Q, * d_uT, * d_R;
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_Q, n * n * sizeof(float));
    cudaMalloc((void**)&d_uT, n * n * sizeof(float));
    cudaMalloc((void**)&d_R, n * n * sizeof(float));

    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDims(BLOCK_SIZE_1);
    dim3 gridDims(ceil((float)n / blockDims.x));

    // Copy A to uT
    cudaMemcpy(d_uT, d_A, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

    // Normalize Q
    normalize << <gridDims, blockDims >> > (d_uT, d_Q, n);

    // Compute uT
    for (int i = 1; i < n; i++) {
        cudaMemcpy(d_uT + i * n, d_A + i * n, n * sizeof(float), cudaMemcpyDeviceToDevice);

        for (int j = 0; j < i; j++) {
            computeDotProduct << <gridDims, blockDims >> > (d_A, d_Q, d_uT, n, i, j);
        }

        normalizeU << <gridDims, blockDims >> > (d_uT, d_Q, n, i);
    }

    // Compute R
    dim3 BlockDims2D(BLOCK_SIZE_2, BLOCK_SIZE_2);
    dim3 GridDims2D(ceil((float)n / BlockDims2D.x), ceil((float)n / BlockDims2D.y));
    /*
    for (int i = 0; i < n; i++) {
        computeDotProductR << <GridDims2D, BlockDims2D >> > (d_A, d_Q, d_R, n, i);
    }
    */
    computeDotProductR << <GridDims2D, BlockDims2D >> > (d_A, d_Q, d_R, n);

    // Transpose Q
    transpose << <GridDims2D, BlockDims2D >> > (d_Q, n);

    cudaMemcpy(Q, d_Q, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(R, d_R, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_Q);
    cudaFree(d_uT);
    cudaFree(d_R);
}


//void testQRMain()
int main()
{
    int size = 2048;
    float* A = new float[size * size];
    float* Q = new float[size * size];
    float* QMul = new float[size * size];
    float* R = new float[size * size];
    float* RMul = new float[size * size];
    float* Q_CUDA = new float[size * size];
    float* R_CUDA = new float[size * size];

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i <= j)
                A[i * size + j] = GEN_FLOAT;
            else
                A[i * size + j] = A[j * size + i];
        }
    }

    DS_timer timer(3);
    timer.setTimerName(0, "QR Serial");
    timer.setTimerName(1, "QR Parallel");
    timer.setTimerName(2, "QR CUDA");

    timer.onTimer(0);
    QR_Decomposition(A, Q, R, size);
    timer.offTimer(0);

    /*
    timer.onTimer(1);
    QR_Decomposition_Multi(A, QMul, RMul, size);
    timer.offTimer(1);
    */


    timer.onTimer(2);
    QR_Decomposition_CUDA(A, Q_CUDA, R_CUDA, size);
    timer.offTimer(2);
    timer.printTimer();

    bool checkQ = true;
    int countDiffQ = 0;
    float maxDiffQ = 0.0;
    int idxQ = 0;
    for (int i = 0; i < size * size; i++)
    {
        //if (abs(Q[i] - QMul[i]) > tol || abs(Q[i] - Q_CUDA[i]) > tol)
        float diff = abs(Q[i] - Q_CUDA[i]);
        if (diff > tol)
        {
            checkQ = false;
            //printf("[%d] Q is not Correct! (%f, %f, %f)\n",i, Q[i], QMul[i], Q_CUDA[i]);
            //printf("[%d] Q is not Correct! (%f, %f)\n",i, Q[i], Q_CUDA[i]);
            //break;
            if (maxDiffQ <= diff)
            {
                maxDiffQ = diff;
                idxQ = i;
            }
            countDiffQ++;
        }
    }

    bool checkR = true;
    int countDiffR = 0;
    float maxDiffR = 0.0;
    int idxR = 0;
    for (int i = 0; i < size * size; i++)
    {
        //if (abs(R[i] - RMul[i]) > tol || abs(R[i] - R_CUDA[i]) > tol)
        float diff = abs(R[i] - R_CUDA[i]);
        if (diff > tol)
        {
            checkQ = false;
            //printf("[%d] R is not Correct! (%f, %f, %f)\n",i, R[i], RMul[i], R_CUDA[i]);
            //printf("[%d] R is not Correct! (%f, %f)\n",i, R[i], R_CUDA[i]);
            //break;
            if (maxDiffR <= diff)
            {
                maxDiffR = diff;
                idxR = i;
            }
            countDiffR++;
        }
    }

    if (checkQ && checkR)
    {
        printf("QR is Correct!\n");
    }
    else
    {
        printf("[Total: %d]\n", size * size);
        printf("Different Q has %d, [%d] MaxDiff: %f, Value(%f, %f)\n", countDiffQ, idxQ, maxDiffQ, Q[idxQ], Q_CUDA[idxQ]);
        printf("Different R has %d, [%d] MaxDiff: %f, Value(%f, %f)\n", countDiffR, idxR, maxDiffR, R[idxR], R_CUDA[idxR]);
    }

    delete A, Q, R, QMul, RMul, Q_CUDA, R_CUDA;

    return 0;
}

void eigenTestMain()
//int main()
{
    int size = 256;
    float* A = new float[size * size];
    float* eigVal = new float[size];
    float* eigValMul = new float[size];
    float* eigVec = new float[size * size];
    float* eigVecMul = new float[size * size];

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i <= j)
                A[i * size + j] = GEN_FLOAT;
            else
                A[i * size + j] = A[j * size + i];
        }
    }

    DS_timer timer(2);
    timer.setTimerName(0, "QR Eigen Serial");
    timer.setTimerName(1, "QR Eigen Parallel");

    timer.onTimer(0);
    findEigenQR(A, eigVal, eigVec, size, size);
    timer.offTimer(0);

    timer.onTimer(1);
    findEigenQR_Multi(A, eigValMul, eigVecMul, size, size);
    timer.offTimer(1);
    timer.printTimer();

    bool isEigvalSame = true;
    for (int i = 0; i < size; i++)
    {
        if (eigVal[i] != eigValMul[i])
        {
            isEigvalSame = false;
            printf("[%d] Serial: %f, Parallel: %f\n", i, eigVal[i], eigValMul[i]);
            break;
        }
    }

    bool isEigvecSame = true;
    for (int i = 0; i < size * size; i++)
    {
        if (eigVec[i] != eigVecMul[i])
        {
            isEigvecSame = false;
            printf("[%d] Serial: %f, Parallel: %f\n", i, eigVal[i], eigValMul[i]);
            break;
        }
    }

    if (isEigvalSame)
    {
        printf("Eigen Value is Same!\n");
    }
    else
    {
        printf("Eigen Value is not Same!\n");
    }
    if (isEigvecSame)
    {
        printf("Eigen Vector is Same!\n");
    }
    else
    {
        printf("Eigen Vector is not Same!\n");
    }

    delete A, eigVal, eigValMul, eigVec, eigVecMul;
}