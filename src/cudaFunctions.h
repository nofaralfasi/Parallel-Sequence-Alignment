#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

#define THREADS_PER_BLOCK 1024
#define THREADS_2D 32

__global__ void computeScoringMatrix(float *scoringMatrix_C, int len1, int len2, int maxN, char *seq1, char *seq2, float weights[NUM_WEIGHTS]);
__device__ char *_strchr(const char *s, int c);
__device__ float cudaComputePairwiseComparison(char c1, char c2, float weights[NUM_WEIGHTS]);
__device__ int cudaCheckIfInGroup(const char *group[MAX_LEN_GROUP_SEQ], int groupLen, char c1, char c2);

#endif /* CUDAFUNCTIONS_H_ */
