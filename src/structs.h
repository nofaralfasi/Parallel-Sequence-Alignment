#ifndef STRUCTS_H_
#define STRUCTS_H_

#define OMP_PART 2
#define NUM_WEIGHTS 4
#define NUM_CONSERV_GROUPS 9
#define NUM_SEMI_CONSERV_GROUPS 11
#define MAX_LEN_SEQ1 3000
#define MAX_LEN_SEQ2 2000
#define MAX_LEN_GROUP_SEQ 7
#define PENALTY -2
#define GAP 2
#define OMP_NUM_THREADS 4096
#define TRUE 1
#define FALSE 0
#define ERROR -1

struct Seq2ResultStruct {
	int n; // offset
	int k; // location of hyphen sign
}typedef Seq2Result;

struct InputDataStruct {
	float weightsArray[NUM_WEIGHTS];
	char *seq1;
	int numOfSequences2;
	int *seq2Lens;
	int maxSeq2Len;
	char **seq2Array;
}typedef InputData;

#endif /* STRUCTS_H_ */
