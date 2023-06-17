#include "DataProcessing.h"
#include <stdio.h>

using namespace std;


bool getData(FILE* fp, float* point_x, float* point_y, int n) {
	if (fp == NULL) {
		printf("FILE NOT OPEN!\n");
		return false;
	}

	int i = 0;
	while (!feof(fp)) {
		double x = 0;
		double y = 0;

#ifdef _WIN64
		fscanf_s(fp, "%lf %lf", &x, &y);
#else
		fscanf(fp, "%lf %lf", &x, &y);
#endif
		point_x[i] = x;
		point_y[i] = y;
		i++;
	}

	fclose(fp);

	if (i == n)
		return true;

	return false;
}

void saveData(const char* fileName, float* point_x, float* point_y, int* results, int n) {
	FILE* fp;
#ifdef _WIN64
	fopen_s(&fp, fileName, "w");
#else
	fp = fopen(fileName, "w");
#endif


	if (fp == NULL) {
		printf("FILE NOT OPEN!\n");
		return;
	}

	for (int i = 0; i < n; i++)
		fprintf(fp, "%lf %lf %d\n", point_x[i], point_y[i], results[i]);

	fclose(fp);
}


void data_main()
//int main() 
{
	const char* input_path = "circle_data.txt";
	int n = 10000;
	FILE* fp;
#ifdef _WIN64
	fopen_s(&fp, input_path, "r");
#else
	fp = fopen(input_path, "r");
#endif
	//Point* points = new Point[n];
	//getData(fp, points, n);
	//for (int i = 0; i < n; i++)
	//	printf("%lf %lf\n", points[i].x, points[i].y);
}