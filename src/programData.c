#include "programData.h"

InputData readInputFile(const char *filename) {
	int i, seqLength, numSequences;
	char seq[MAX_LEN_SEQ1];
	FILE *f;
	InputData inputData;

	f = fopen(filename, "r");
	if (!f) // (f == NULL) --> file opening didn't succeed
		exit(1);

	for (i = 0; i < NUM_WEIGHTS; i++) {
		fscanf(f, "%f", &inputData.weightsArray[i]);
	}

	fscanf(f, "%s", seq);
	seqLength = strlen(seq);
	inputData.seq1 = (char *) calloc(seqLength, sizeof(char));
	if (!inputData.seq1) // (seqContent == NULL) --> allocation didn't succeed
	{
		printf("ERROR during seq1.seqContent memory allocating!\n");
		exit(1);
	}

	strcpy(inputData.seq1, seq);

	fscanf(f, "%d", &numSequences);
	inputData.numOfSequences2 = numSequences;
	inputData.seq2Array = (char **) calloc(numSequences, sizeof(char*));
	inputData.seq2Lens = (int *) calloc(numSequences, sizeof(int));
	if (!inputData.seq2Array || !inputData.seq2Lens) {
		printf("ERROR during seq2Array or seq2Lens memory allocating!\n");
		exit(1);
	}

	inputData.maxSeq2Len = 0;

	for (i = 0; i < numSequences; i++) {
		fscanf(f, "%s", seq);
		seqLength = strlen(seq);
		inputData.seq2Array[i] = (char *) calloc(seqLength, sizeof(char));
		if (!inputData.seq2Array[i]) // (seqContent == NULL) --> allocation didn't succeed
		{
			printf("ERROR during seq2.seqContent memory allocating!\n");
			exit(1);
		}

		inputData.seq2Lens[i] = seqLength;
		if (seqLength > inputData.maxSeq2Len) {
			inputData.maxSeq2Len = seqLength;
		}

		strcpy(inputData.seq2Array[i], seq);
	}

	fclose(f);

	updateWeightsAccordingToScoreFunction(inputData.weightsArray);

	return inputData;
}

void updateWeightsAccordingToScoreFunction(float weights[NUM_WEIGHTS]) {
	int i;
	for (i = 1; i < NUM_WEIGHTS; i++) {
		weights[i] = (-1) * weights[i];
	}
}

void sendInputDataToOtherProcess(InputData *inputData, int numSeq2ToSend, int dest, int tag) {
	int i;

	// sends weights
	MPI_Send(inputData->weightsArray, NUM_WEIGHTS, MPI_FLOAT, dest, tag,
	MPI_COMM_WORLD);
	// sends length of seq1
	int len = strlen(inputData->seq1);
	MPI_Send(&len, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
	// sends seq1
	MPI_Send(inputData->seq1, len, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	// sends number of seq2
	MPI_Send(&inputData->numOfSequences2, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
	// sends maximum length of seq2
	MPI_Send(&inputData->maxSeq2Len, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
	// sends lengths of seq2
	MPI_Send(inputData->seq2Lens + numSeq2ToSend, inputData->numOfSequences2 / 2,
	MPI_INT, dest, tag, MPI_COMM_WORLD);
	// sends seq2
	for (i = numSeq2ToSend; i < inputData->numOfSequences2; i++)
		MPI_Send(inputData->seq2Array[i], inputData->seq2Lens[i], MPI_CHAR, dest, tag, MPI_COMM_WORLD);
}

void receiveInputDataFromMainProcess(InputData *inputData, int *numSeq2ToReceive, int source, int tag) {
	int i, len;
	MPI_Status status;

	// receives weights
	MPI_Recv(inputData->weightsArray, NUM_WEIGHTS, MPI_FLOAT, source, tag,
	MPI_COMM_WORLD, &status);
	// receives length of seq1
	MPI_Recv(&len, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	// allocates memory for seq1 and than receives it
	inputData->seq1 = (char*) calloc(len, sizeof(char));
	MPI_Recv(inputData->seq1, len, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
	// receives number of seq2
	MPI_Recv(&inputData->numOfSequences2, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	// receives maximum length of seq2
	MPI_Recv(&inputData->maxSeq2Len, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

	*numSeq2ToReceive = (inputData->numOfSequences2 / 2);
	// allocates memory for receiving next arrays
	inputData->seq2Array = (char**) calloc(*numSeq2ToReceive, sizeof(char*));
	inputData->seq2Lens = (int*) calloc(*numSeq2ToReceive, sizeof(int));
	if (!inputData->seq2Array || !inputData->seq2Lens) {
		printf("ERROR during seq2Array or seq2Lens memory allocating!\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}
	// receives lengths of seq2
	MPI_Recv(inputData->seq2Lens, *numSeq2ToReceive, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
	// receives seq2 (half of them)
	for (i = 0; i < *numSeq2ToReceive; i++) {
		inputData->seq2Array[i] = (char*) calloc(inputData->seq2Lens[i], sizeof(char));
		MPI_Recv(inputData->seq2Array[i], inputData->seq2Lens[i], MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
	}
}

void writeOutputFile(const char *filename, int numOfSeq2, Seq2Result *seq2Results) {
	int i;
	FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("Failed opening output file. Exiting!\n");
		exit(1);
	}

	for (i = 0; i < numOfSeq2; i++) {
		fprintf(f, "%d %d\n", seq2Results[i].n, seq2Results[i].k);
	}
	fclose(f);
}

void printData(InputData inputData, int arraysLen) {
	int i;
	printf("\nWeights: ");
	for (i = 0; i < NUM_WEIGHTS; i++)
		printf("%.2f, ", inputData.weightsArray[i]);
	printf("\nseq1: %s\n", inputData.seq1);
	printf("numOfSequences2: %d\nSequences2: ", inputData.numOfSequences2);
	for (i = 0; i < arraysLen; i++)
		printf("%s, ", inputData.seq2Array[i]);
	printf("\nseq2Lens: ");
	for (i = 0; i < arraysLen; i++)
		printf("%d, ", inputData.seq2Lens[i]);
	printf("\n");
}

void freeData(InputData inputData, int arraysLen) {
	int i;

	free(inputData.seq1);
	free(inputData.seq2Lens);
	for (i = 0; i < arraysLen; i++)
		free(inputData.seq2Array[i]);

	free(inputData.seq2Array);
}
