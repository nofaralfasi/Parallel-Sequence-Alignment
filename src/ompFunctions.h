#ifndef OMPFUNCTIONS_H_
#define OMPFUNCTIONS_H_

#include "calculations.h"

void calculateResultsWithOpenMP(Seq2Result *seq2Results, InputData *inputData, int numOfSeq2);
Seq2Result calculateScoringMatrix(char *seq1, char *seq2, float weights[NUM_WEIGHTS]);
float computePairwiseComparison(char c1, char c2, float weights[NUM_WEIGHTS]);
int checkIfInGroup(const char *group[MAX_LEN_GROUP_SEQ], int groupLen, char c1, char c2);
int checkIfInGroupOMP(const char *group[MAX_LEN_GROUP_SEQ], int groupLen, char c1, char c2);

#endif /* OMPFUNCTIONS_H_ */
