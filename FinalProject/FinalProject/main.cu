#include <stdio.h>
#include <omp.h>
#include <Eigen/Dense>
#include "AffinityMatrix.cuh"
#include "LaplacianMatrix.cuh"
#include "DataProcessing.h"
#include "DS_definitions.h"
#include "DS_timer.h"
#include <utility>
#include <vector>


#include <cusolverDn.h>

using namespace Eigen;
using namespace std;

#define TOL 1e-6

void swap(double& x1, double& x2) {
	double temp = x1;
	x1 = x2;
	x2 = temp;
}

void labelDecomposition(MatrixXd& eigenVectors, int* results, int n) {
	std::vector<std::pair<double, int>> dictionary;
	int size = 0;
	for (int i = 0; i < n; i++) {
		if (dictionary.empty()) {
			std::pair<double, int> item(eigenVectors(i, 1), 1);
			dictionary.push_back(item);
			size++;
		}
		else {
			for (int j = 0; j < size; j++) {
				if (abs(dictionary[j].first - eigenVectors(i, 1)) < 1e-3) {
					dictionary[j].second += 1;
					break;
				}
				if (j == size - 1) {
					std::pair<double, int> item(eigenVectors(i, 1), 1);
					dictionary.push_back(item);
					size++;
				}
			}
		}
	}

	for (int i = 0; i < 2; i++) {
		int max = i;
		for (int j = i + 1; j < size; j++) {
			if (dictionary[max].second < dictionary[j].second) {
				max = j;
			}
		}
		std::swap(dictionary[max], dictionary[i]);
	}

	double labels[2];
	labels[0] = dictionary[0].first;
	labels[1] = dictionary[1].first;

	//printf("Size = %d\n", size);
	//printf("label 1 %lf\nlabel 2 %lf\n", labels[0], labels[1]);

	for (int i = 0; i < n; i++) {
		if (abs(eigenVectors(i, 1) - labels[0]) > abs(eigenVectors(i, 1) - labels[1])) {
			results[i] = 1;
		}
		else {
			results[i] = -1;
		}
	}
}

void find_2nd_Min(VectorXd& eigenValue, MatrixXd& eigenVector) {
	int n = eigenVector.rows();

	for (int i = 0; i < 2; i++) {
		int min = i;
		for (int j = 1; j < n; j++) {
			if (eigenValue[min] > eigenValue[j]) {
				min = j;
			}
		}
		swap(eigenValue[min], eigenValue[i]);
		for (int j = 0; j < n; j++) {
			swap(eigenVector(j, min), eigenVector(j, i));
		}
	}
}

int realMain(int argc, char** argv)
//int main(int argc, char** argv)
{
	string fname = argv[1];
	string file_name = ".\\data\\" + fname;
	int n = atoi(argv[2]);
	FILE* fp;
#ifdef _WIN64
	fopen_s(&fp, file_name.c_str(), "r");
#else
	fp = fopen(file_name.c_str(), "r");
#endif
	float* x = new float[n];
	float* y = new float[n];

	bool isOpen = getData(fp, x, y, n);
	if (!isOpen) {
		return -1;
	}

	string name[2] = { "Single Method", "Multi Method" };
	DS_timer timer(2);
	for (int i = 0; i < 2; i++)
		timer.setTimerName(i, name[i]);

	float* resultAffinSingle = new float[n * n];
	generateAffinityMatrix(x, y, n, resultAffinSingle);
	float* serialResult = generateLaplacianMatrix(resultAffinSingle, n);

	MatrixXd A(n, n);
	for (int i = 0; i < n * n; i++) {
		A(i / n, i % n) = serialResult[i];
	}

	MatrixXd eigenVectors(n, n);
	VectorXd eigenValues(n);
	EigenSolver<MatrixXd> solver = EigenSolver<MatrixXd>(A);

	MatrixXcd solveEigenVector = solver.eigenvectors();
	VectorXcd solveEigenValue = solver.eigenvalues();

	for (int i = 0; i < n; i++) {
		eigenValues[i] = solveEigenValue[i].real();
		for (int j = 0; j < n; j++) {
			eigenVectors(i, j) = solveEigenVector(i, j).real();
		}
	}

	int* results = new int[n];
	find_2nd_Min(eigenValues, eigenVectors);
	labelDecomposition(eigenVectors, results, n);




	bool isCorrect = true;
	int idx[2] = { 0 };
	/*
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double single = dataSingle[i][j];
			if (abs(single - dataMuliti1[i][j]) > TOL || abs(single - dataMuliti2[i][j]) > TOL || abs(single - dataMuliti3[i][j]) > TOL
				|| abs(single - dataMuliti4_1[i][j]) > TOL || abs(single - dataMuliti4_2[i][j]) > TOL) {
				isCorrect = false;
				idx[0] = i; idx[1] = j;
				break;
			}
		}
	}
	*/
	/*
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double single = dataSingle[i][j];
			if (abs(single - dataMuliti1[i][j]) > TOL) {
				isCorrect = false;
				idx[0] = i; idx[1] = j;
				break;
			}
		}
	}
	*/
	if (isCorrect) {
		printf("Data is correct.\n");
	}
	else {
		printf("Data is not correct. ");
		//printf("Single[%d][%d] : %lf\n", idx[0], idx[1], dataSingle[idx[0]][idx[1]]);
		//printf("dataMuliti[%d][%d] : %lf\n", idx[0], idx[1], dataMuliti1[idx[0]][idx[1]]);
		/*
		printf("dataMuliti2[%d][%d] : %lf\n", idx[0], idx[1], dataMuliti2[idx[0]][idx[1]]);
		printf("dataMuliti3[%d][%d] : %lf\n", idx[0], idx[1], dataMuliti3[idx[0]][idx[1]]);
		printf("dataMuliti4_1[%d][%d] : %lf\n", idx[0], idx[1], dataMuliti4_1[idx[0]][idx[1]]);
		printf("dataMuliti4_2[%d][%d] : %lf\n", idx[0], idx[1], dataMuliti4_2[idx[0]][idx[1]]);
		*/
	}

	for (int i = 0; i < 8; i++)
	{
		file_name.pop_back();
	}

	string version[] = { "SingleMethod", "MultiMethod" };
	timer.printTimer();

	string saveName;
	saveName = file_name + "Serial_result.txt";
	saveData(saveName.c_str(), x, y, results, n);


	delete[] x, y, resultAffinSingle, results;

	return 0;
}