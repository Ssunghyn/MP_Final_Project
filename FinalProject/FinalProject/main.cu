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

#include <cuda_runtime.h>

using namespace Eigen;
using namespace std;

#define TOL 1e-1

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

	for (int i = 0; i < n; i++) {
		if (abs(eigenVectors(i, 1) - labels[0]) > abs(eigenVectors(i, 1) - labels[1])) {
			results[i] = 1;
		}
		else {
			results[i] = -1;
		}
	}
}

void labelDecomposition(float* eigenVectors, int* results, int n) {
	std::vector<std::pair<double, int>> dictionary;
	int size = 0;
	for (int i = 0; i < n; i++) {
		if (dictionary.empty()) {
			std::pair<double, int> item(eigenVectors[i * n + 1], 1);
			dictionary.push_back(item);
			size++;
		}
		else {
			for (int j = 0; j < size; j++) {
				if (abs(dictionary[j].first - eigenVectors[i * n + 1]) < 1e-3) {
					dictionary[j].second += 1;
					break;
				}
				if (j == size - 1) {
					std::pair<double, int> item(eigenVectors[i * n + 1], 1);
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
		if (abs(eigenVectors[i * n + 1] - labels[0]) > abs(eigenVectors[i * n + 1] - labels[1])) {
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

void find_2nd_Min(float* eigenValue, float* eigenVector, int n) {
	for (int i = 0; i < 2; i++) {
		int min = i;
		for (int j = 1; j < n; j++) {
			if (eigenValue[min] > eigenValue[j]) {
				min = j;
			}
		}
		swap(eigenValue[min], eigenValue[i]);
		for (int j = 0; j < n; j++) {
			swap(eigenVector[min * n + j], eigenVector[i * n + j]);
		}
	}
}

//int realMain(int argc, char** argv)
int main(int argc, char** argv)
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

	string name[2] = { "Single Method", "Parallel Method" };
	DS_timer timer(2);
	for (int i = 0; i < 2; i++)
		timer.setTimerName(i, name[i]);


	// ***************** SERIAL START ********************//
	float* resultAffinSingle = new float[n * n];
	MatrixXd A(n, n);
	MatrixXd eigenVectorSerial(n, n);
	VectorXd eigenValueSerial(n);
	int* resultsSerial = new int[n];


	timer.onTimer(0);
	generateAffinityMatrix(x, y, n, resultAffinSingle);
	float* serialLaplacian = generateLaplacianMatrix(resultAffinSingle, n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			A(i, j) = serialLaplacian[i * n + j];
	}

	SelfAdjointEigenSolver<MatrixXd> solver = SelfAdjointEigenSolver<MatrixXd>(A);

	MatrixXcd solveEigenVector = solver.eigenvectors();
	VectorXcd solveEigenValue = solver.eigenvalues();

	for (int i = 0; i < n; i++) {
		eigenValueSerial[i] = solveEigenValue[i].real();
		for (int j = 0; j < n; j++) {
			eigenVectorSerial(i, j) = solveEigenVector(i, j).real();
		}
	}
	timer.offTimer(0);
	find_2nd_Min(eigenValueSerial, eigenVectorSerial);
	labelDecomposition(eigenVectorSerial, resultsSerial, n);
	
	// ***************** SERIAL END ********************//


	// ***************** PARALLEL START ********************//
	float* resultAffinMulti = new float[n * n];
	int* resultsParallel = new int[n];
	float* d_result = NULL;
	MatrixXd B(n, n);
	MatrixXd eigenVectorParallel(n, n);
	VectorXd eigenValueParallel(n);

	timer.onTimer(1);

	generateAffinityMatrix_cuda(x, y, n, resultAffinMulti, d_result);
	float* parallelLaplacian = generateLaplacianMatrixOmp(resultAffinMulti, n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			B(i, j) = parallelLaplacian[i * n + j];
	}

	SelfAdjointEigenSolver<MatrixXd> solverParallel = SelfAdjointEigenSolver<MatrixXd>(B);

	MatrixXcd solveEigenVectorParallel = solverParallel.eigenvectors();
	VectorXcd solveEigenValueParallel = solverParallel.eigenvalues();

	for (int i = 0; i < n; i++) {
		eigenValueParallel[i] = solveEigenValueParallel[i].real();
		for (int j = 0; j < n; j++) {
			eigenVectorParallel(i, j) = solveEigenVectorParallel(i, j).real();
		}
	}
	timer.offTimer(1);

	find_2nd_Min(eigenValueParallel, eigenVectorParallel);
	labelDecomposition(eigenVectorParallel, resultsParallel, n);

	

	// ***************** PARALLEL END ********************//


	bool isCorrect = true;
	int idx = 0;

	for (int i = 0; i < n; i++) {
		if (abs(serialLaplacian[i] - parallelLaplacian[i]) > TOL) {
			isCorrect = false;
			idx = i;
			break;
		}
	}

	if (isCorrect) {
		printf("Data is correct.\n");
	}
	else {
		printf("Data is not correct. ");
		printf("Serialize[%d] : %f\n", idx, eigenValueSerial[idx]);
		printf("dataMuliti[%d] : %f\n", idx, eigenValueParallel[idx]);
	}

	for (int i = 0; i < 8; i++)
	{
		file_name.pop_back();
	}

	string version[] = { "SerialSpectral", "ParallelSpectral" };
	timer.printTimer();

	string saveName;
	saveName = file_name + version[0] + "_result.txt";
	saveData(saveName.c_str(), x, y, resultsSerial, n);
	saveName = file_name + version[1] + "_result.txt";
	saveData(saveName.c_str(), x, y, resultsParallel, n);

	delete[] x, y, resultAffinSingle, resultAffinMulti, eigenValueParallel, eigenVectorParallel;

	return 0;
}