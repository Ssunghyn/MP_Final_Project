#pragma once
#include <stdio.h>


bool getData(FILE* fp, float* point_x, float* point_y, int n);
void saveData(const char* fileName, float* point_x, float* point_y, int* results, int n);