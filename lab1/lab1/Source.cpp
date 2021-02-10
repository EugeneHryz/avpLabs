#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>

#define ROWS 8
#define COLUMNS ROWS

using namespace std;

// creates and optionally initializes internal matrix
float** createMatrix(int rows, int columns, bool fillWithZeros) {

	float** matrix = nullptr;
	matrix = new float*[rows];

	for (int i = 0; i < rows; i++) {

		matrix[i] = new float[columns];

		for (int j = 0; j < columns; j++) {

			float randFloat = 0.0;
			if (!fillWithZeros) {
				randFloat = rand();
				randFloat /= RAND_MAX;
				randFloat *= 100;
			}

			matrix[i][j] = randFloat;
		}
	}
	return matrix;
}

void freeMatrix(float** matrix, int rows) {

	for (int i = 0; i < rows; i++) {

		delete[] matrix[i];
	}
	delete[] matrix;
}

void printMatrix(float** matrix, int rows, int columns) {

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < columns; j++) {

			cout << setw(20) << matrix[i][j];
		}
		cout << "\n";
	}
}

// creates and initializes big matrix
float**** createBigMatrix(int rows, int columns, bool fillWithZeros) {

	float**** matrix = nullptr;
	matrix = new float***[rows];

	for (int i = 0; i < rows; i++) {

		matrix[i] = new float**[columns];
		for (int j = 0; j < columns; j++) {

			matrix[i][j] = createMatrix(ROWS, COLUMNS, fillWithZeros);
		}
	}
	return matrix;
}

void printBigMatrix(float**** matrix, int rows, int columns) {

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < ROWS; j++) {

			for (int k = 0; k < columns; k++) {

				for (int l = 0; l < COLUMNS; l++) {

					cout << setw(20) << matrix[i][k][j][l];
				}
				cout << "   ";
			}
			cout << "\n";
		}
		cout << "\n\n";
	}
}

void freeBigMatrix(float**** matrix, int rows, int columns) {

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < columns; j++) {

			freeMatrix(matrix[i][j], ROWS);
		}
	}
}

float**** multiplyMatrices(float**** matrixA, float**** matrixB, int rowsA, int columnsA, int rowsB, int columnsB) {

	float**** resultMatrix = nullptr;

	if (columnsA == rowsB) {

		resultMatrix = createBigMatrix(rowsA, columnsB, true);

		std::chrono::high_resolution_clock::time_point startPoint =
			std::chrono::high_resolution_clock::now();

		for (int i = 0; i < rowsA; i++) {

			for (int j = 0; j < columnsB; j++) {

				for (int k = 0; k < columnsA; k++) {

					for (int l = 0; l < ROWS; l++) {

						for (int m = 0; m < COLUMNS; m++) {

							for (int n = 0; n < COLUMNS; n++) {

								resultMatrix[i][j][l][m] += matrixA[i][k][l][n] * matrixB[k][j][n][m];
							}
						}
					}
				}
			}
		}

		std::chrono::high_resolution_clock::time_point endPoint =
			std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> duration =
			std::chrono::duration_cast<std::chrono::duration<double>>(endPoint - startPoint);

		cout << "\nTime passed: " << duration.count() << " seconds\n";
	}
	else {
		cout << "Matrices cannot be multiplied.\n";
	}
	return resultMatrix;
}

float**** multiplyMatricesUsingIntrinsics(float**** matrixA, float**** matrixB, int rowsA, int columnsA, int rowsB, int columnsB) {

	float**** matrixC = nullptr;

	if (columnsA == rowsB) {

		matrixC = createBigMatrix(rowsA, columnsB, true);

		std::chrono::high_resolution_clock::time_point startPoint =
			std::chrono::high_resolution_clock::now();

		for (int i = 0; i < rowsA; i++) {

			for (int j = 0; j < columnsB; j++) {

				for (int k = 0; k < columnsA; k++) {

					for (int l = 0; l < ROWS; l++) {

						for (int m = 0; m < COLUMNS; m++) {

							__m256 elementA = _mm256_set1_ps(matrixA[i][k][l][m]);

							float* tempB = matrixB[k][j][m];
							float* tempC = matrixC[i][j][l];

							__m256 tempB_256 = _mm256_load_ps(tempB);
							__m256 tempC_256 = _mm256_load_ps(tempB);

							__m256 mulResult = _mm256_mul_ps(tempB_256, tempC_256);
							__m256 newC = _mm256_mul_ps(mulResult, tempC_256);

							_mm256_store_ps(tempC, newC);
						}
					}
				}
			}
		}

		std::chrono::high_resolution_clock::time_point endPoint =
			std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> duration =
			std::chrono::duration_cast<std::chrono::duration<double>>(endPoint - startPoint);

		cout << "\nTime passed: " << duration.count() << " seconds\n";
	}
	else {
		cout << "Matrices cannot be multiplied.\n";
	}
	return matrixC;
}


int main() {

	const int rowsA = 100, columnsA = 100;

	float**** matrixA = createBigMatrix(rowsA, columnsA, false);

	const int rowsB = 100, columnsB = 100;
	float**** matrixB = createBigMatrix(rowsB, columnsB, false);

	float**** matrixC1 = multiplyMatrices(matrixA, matrixB, rowsA, columnsA, rowsB, columnsB);

	cout << "\nManual Vectorization:\n";

	float**** matrixC2 = multiplyMatricesUsingIntrinsics(matrixA, matrixB, rowsA, columnsA, rowsB, columnsB);

	freeBigMatrix(matrixA, rowsA, columnsA);
	freeBigMatrix(matrixB, rowsB, columnsB);
	freeBigMatrix(matrixC1, rowsA, columnsB);
	freeBigMatrix(matrixC2, rowsA, columnsB);

	return 0;
}
