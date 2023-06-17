#include "DS_definitions.h"
#include "DS_timer.h"
#include "LaplacianMatrix.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>

#define Genfloat ((float)(rand() % 100) / 100.0)
#define N 40000
#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))

bool checkResult(float* c, float* dc)
{
	bool result = true;
	for (int i = 0; i < N * N; i++)
	{
		if (c[i] != dc[i])
		{
			printf("Result is wrong at index : [%d][%d], %f, %f\n", i / N, i - (i / N * N), c[i], dc[i]);
			result = false;
		}
	}
	if (result) printf("Result is correct!\n");
	return result;
}

int mainLaplacianTest() 
//int main()
{
	float *a, *b, *c, *d;
	a = new float[N * N];
	b = new float[N * N];
	c = new float[N * N];
	d = new float[N * N];
	cudaFree(0);
	float* r1, * r2, * r3, * r4;
#pragma omp parallel num_threads(1) 
	{
		printf("\n");
	}
	LOOP_I(N) {
		for (int j = i; j < N; j++) {
			a[ID2INDEX(i, j, N)] = Genfloat;
			a[ID2INDEX(j, i, N)] = a[ID2INDEX(i, j, N)];
			b[ID2INDEX(i, j, N)] = a[ID2INDEX(i, j, N)];
			b[ID2INDEX(j, i, N)] = a[ID2INDEX(i, j, N)];
			c[ID2INDEX(i, j, N)] = a[ID2INDEX(i, j, N)];
			c[ID2INDEX(j, i, N)] = a[ID2INDEX(i, j, N)];
			d[ID2INDEX(i, j, N)] = a[ID2INDEX(i, j, N)];
			d[ID2INDEX(j, i, N)] = a[ID2INDEX(i, j, N)];
		}
	}
	DS_timer timer(4, 1);

	timer.setTimerName(0, "Serial");
	timer.setTimerName(1, "Cuda & OMP");
	timer.setTimerName(2, "Stream Cuda & OMP");
	timer.setTimerName(3, "OMP");

	timer.onTimer(0);
	r1 = generateLaplacianMatrix(a, N);
	timer.offTimer(0);
	
	timer.onTimer(1);
	r2 = generateLaplacianMatrixParallel(b, N);
	timer.offTimer(1);

	timer.onTimer(2);
	r3 = generateLaplacianMatrixParallel2(c, N);
	timer.offTimer(2);

	timer.onTimer(3);
	r4 = generateLaplacianMatrixOmp(d, N);
	timer.offTimer(3);

	//printf("�ø���� �⺻ cuda ��\n");
	//checkResult(r1, r2);
	//printf("�ø���� Stream cuda ��\n");
	//checkResult(r1, r3);
	timer.printTimer();
}