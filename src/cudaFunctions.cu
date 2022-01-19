#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "calculations.h"
#include "cudaFunctions.h"

// Kernel
__global__ void computeScoringMatrix(float *scoringMatrix_C, int len1, int len2, int maxN, char *seq1, char *seq2, float weights[NUM_WEIGHTS]) {
	int i = blockDim.y * blockIdx.y + threadIdx.y; // row
	int j = blockDim.x * blockIdx.x + threadIdx.x; // column

	if (i < len2) {
		if (j < i)
			scoringMatrix_C[j + 1 + (i + 1) * len1] = PENALTY * weights[0];
		else if (j >= i && j <= maxN + i)
			scoringMatrix_C[j + 1 + (i + 1) * len1] = cudaComputePairwiseComparison(seq1[j], seq2[i], weights);
	}
}

__device__ float cudaComputePairwiseComparison(char c1, char c2, float weights[NUM_WEIGHTS]) {
	const char *CONSERV_GROUPS[NUM_CONSERV_GROUPS] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" }; //':'
	const char *SEMI_CONSERV_GROUPS[NUM_SEMI_CONSERV_GROUPS] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY",
			"FVLIM" }; //'.'

	if (c1 == c2)
		return weights[0];
	else if (cudaCheckIfInGroup(CONSERV_GROUPS, NUM_CONSERV_GROUPS, c1, c2))
		return weights[1];
	else if (cudaCheckIfInGroup(SEMI_CONSERV_GROUPS, NUM_SEMI_CONSERV_GROUPS, c1, c2))
		return weights[2];
	else
		return weights[3];
}

__device__ int cudaCheckIfInGroup(const char *group[MAX_LEN_GROUP_SEQ], int groupLen, char c1, char c2) {
	int m;
	for (m = 0; m < groupLen; m++) {
		if (_strchr(group[m], c1) && _strchr(group[m], c2))
			return TRUE;
	}
	return FALSE;
}

// Checks if char c is in string s on GPU device
__device__ char *_strchr(const char *s, int c) {
	while (*s != (char) c)
		if (!*s++)
			return 0;
	return (char *) s;
}

// Host function that copies the data and launches the work on GPU device
__host__ void calculateScoringMatrixOnGPU(Seq2Result *seq2Results, InputData *inputData, int totalNumOfSeq2, int cudaNumSeq2) {
	int i, j, len1, len2, maxN;
	float *weights_C, minW = inputData->weightsArray[1], penalty = PENALTY * inputData->weightsArray[0], temp;
	float *scoringMatrix_C, **scoringMatrix, **sumMatrix;
	char *seq1_C, *seq2_C;
	dim3 blocksPerGrid, threadsPerBlock(THREADS_2D, THREADS_2D);
	size_t size;
	cudaError_t err = cudaSuccess;

	// Allocate memory on GPU to copy the data from the host
	// 1. weights
	size = NUM_WEIGHTS * sizeof(float);
	err = cudaMalloc((void **) &weights_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory (weights) - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
	// Copy data from host to the GPU memory
	err = cudaMemcpy(weights_C, inputData->weightsArray, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy data from host to device (weights) - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

	// 2. seq1
	len1 = strlen(inputData->seq1) + 1;
	size = len1 * sizeof(char);
	err = cudaMalloc((void **) &seq1_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory (seq1) - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
	// Copy seq1 from host to the GPU memory
	err = cudaMemcpy(seq1_C, inputData->seq1, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy data from host to device (seq1) - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

	// Find minimum weight
	for (i = 2; i < NUM_WEIGHTS; i++) {
		temp = inputData->weightsArray[i];
		if (minW < temp)
			minW = temp;
	}

	// 3. seq2 array
	len2 = (inputData->maxSeq2Len) + 1;
	size = len2 * sizeof(char);
	err = cudaMalloc((void **) &seq2_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory (seq2_C) - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

	// Setting scoringMatrix_C for kernel computation
	size = len1 * len2 * sizeof(float);
	err = cudaMalloc((void **) &scoringMatrix_C, size);
	if (err != cudaSuccess) {
		printf("ERROR during scoringMatrix_C memory allocating - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

	size = len1 * sizeof(float);
	scoringMatrix = (float **) calloc(len2, sizeof(float *));
	sumMatrix = (float **) calloc(len2, sizeof(float *));

	// Runtime Complexity: O(N)
	for (j = 0; j < len2; j++) {
		scoringMatrix[j] = (float *) calloc(len1, sizeof(float));
		sumMatrix[j] = (float *) calloc(len1, sizeof(float));
	}

	//	Loops on each seq2 in array
	for (i = totalNumOfSeq2 - cudaNumSeq2; i < totalNumOfSeq2; i++) {
		len2 = inputData->seq2Lens[i] + 1;
		maxN = len1 - len2;
		size = len2 * sizeof(char);

		// Copy seq2 data from host to the GPU memory
		err = cudaMemcpy(seq2_C, inputData->seq2Array[i], size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy data from host to device (seq2) - %s\n", cudaGetErrorString(err));
			exit (EXIT_FAILURE);
		}

		blocksPerGrid.x = (len1 / THREADS_2D) + 1;
		blocksPerGrid.y = (len2 / THREADS_2D) + 1;

		// Launches kernel
		computeScoringMatrix<<<blocksPerGrid, threadsPerBlock>>>(scoringMatrix_C, len1, len2, maxN, seq1_C, seq2_C, weights_C);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch computeScoringMatrix kernel -  %s\n", cudaGetErrorString(err));
			exit (EXIT_FAILURE);
		}

		size = len1 * sizeof(float);
		for (j = 0; j < len2; j++) {
			err = cudaMemcpy(scoringMatrix[j], (scoringMatrix_C + j * len1), size, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to copy data from device to host (scoringMatrix) - %s\n", cudaGetErrorString(err));
				exit (EXIT_FAILURE);
			}
		}

		scoringMatrix[1][maxN + 1] = penalty;

		computeSumMatrix(sumMatrix, scoringMatrix, len1, len2, minW, penalty);

		computeBestScore(&(seq2Results[i]), scoringMatrix, sumMatrix, len1, len2, minW);

	}

	// Free allocated memory on GPU
	if (cudaFree(weights_C) != cudaSuccess) {
		fprintf(stderr, "Failed to free device weights_C data - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
	if (cudaFree(seq1_C) != cudaSuccess) {
		fprintf(stderr, "Failed to free device seq1_C data - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
	if (cudaFree(seq2_C) != cudaSuccess) {
		fprintf(stderr, "Failed to free device seq2_C data - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
	if (cudaFree(scoringMatrix_C) != cudaSuccess) {
		fprintf(stderr, "Failed to free device scoringMatrix_C data - %s\n", cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

	// Free allocated memory on CPU (host)
	freeMatrix(scoringMatrix, (inputData->maxSeq2Len) + 1);
	freeMatrix(sumMatrix, (inputData->maxSeq2Len) + 1);
}
