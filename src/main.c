#include "programData.h"

// Program main steps:
// 1. Divide the tasks between both processes using MPI
// 1. On each process - perform a first half of its task with OpenMP
// 2. On each process - perform a second half of its task with CUDA
// 3. Collect the result on one of processes
// 4. Write results to output file

int main(int argc, char *argv[]) {
	int size, rank, numOfSeq2, ompNumOfSeq2, cudaNumOfSeq2;
	double time_taken;
	clock_t t_program = clock(), t_omp, t_cuda;
	InputData inputData;
	Seq2Result *seq2Results;
	MPI_Status status;

	// for creating MPI type for results
	MPI_Datatype Seq2Result_MPI_Type;
	int blocklen[2] = { 1, 1 };
	MPI_Datatype type[2] = { MPI_INT, MPI_INT };
	MPI_Aint disp[2] = { offsetof(Seq2Result, k), offsetof(Seq2Result, n) };

	MPI_Init(&argc, &argv);

	// creating MPI type for results
	MPI_Type_create_struct(2, blocklen, disp, type, &Seq2Result_MPI_Type);
	MPI_Type_commit(&Seq2Result_MPI_Type);

	// getting number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != NUM_PROCESSES) {
		printf("Run the program with %d processes only\n", NUM_PROCESSES);
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	// getting number of process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Step 1 - using MPI: divide the tasks between both processes
	if (rank == 0) {
		// get data from input file
		inputData = readInputFile(INPUT_FILE_NAME);

		numOfSeq2 = ceil((double) (inputData.numOfSequences2) / 2);
		sendInputDataToOtherProcess(&inputData, numOfSeq2, 1, 0);

		//allocates memory for seq2Results
		if (!(seq2Results = (Seq2Result*) calloc(inputData.numOfSequences2, sizeof(Seq2Result)))) {
			printf("Error during seq2Results dynamic allocating. Exiting!\n");
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}

	} else {
		receiveInputDataFromMainProcess(&inputData, &numOfSeq2, 0, 0);

		//allocates memory for seq2Results
		if (!(seq2Results = (Seq2Result*) calloc(numOfSeq2, sizeof(Seq2Result)))) {
			printf("Error during seq2Results dynamic allocating. Exiting!\n");
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
	}

	// splits the tasks between OpenMP and CUDA
	ompNumOfSeq2 = numOfSeq2 / OMP_PART;
	cudaNumOfSeq2 = numOfSeq2 - ompNumOfSeq2;


	// Step 1 - using OpenMP: both processes are calculating the results
	t_omp = clock();
	calculateResultsWithOpenMP(seq2Results, &inputData, ompNumOfSeq2);
	// get OpenMP executing time
	t_omp = clock() - t_omp;
	time_taken = ((double) t_omp) / CLOCKS_PER_SEC; // in seconds
	printf("\nOpenMP part took %.2f seconds to execute\n", time_taken);


	// Step 2 - using CUDA: both processes are calculating the results
	t_cuda = clock();
	calculateScoringMatrixOnGPU(seq2Results, &inputData, numOfSeq2, cudaNumOfSeq2);
	// get CUDA executing time
	t_cuda = clock() - t_cuda;
	time_taken = ((double) t_cuda) / CLOCKS_PER_SEC; // in seconds
	printf("\nCUDA part took %.2f seconds to execute\n", time_taken);


	//  Step 3 using MPI - collect the result on one of processes
	if (rank == 0) {
		MPI_Recv(seq2Results + ompNumOfSeq2 + cudaNumOfSeq2, (inputData.numOfSequences2) / 2, Seq2Result_MPI_Type, 1, 0,
				MPI_COMM_WORLD, &status);

		// // Step 4 - writing results to output file
		writeOutputFile(OUTPUT_FILE_NAME, inputData.numOfSequences2, seq2Results);
		printf("\nWrote results to output file!");

		// Calculate the time taken for program to run
		t_program = clock() - t_program;
		time_taken = ((double) t_program) / CLOCKS_PER_SEC; // in seconds
		printf("\nProgram took %.2f seconds to execute\n", time_taken);

		freeData(inputData, inputData.numOfSequences2);
	}
	else {
		MPI_Send(seq2Results, (inputData.numOfSequences2) / 2, Seq2Result_MPI_Type, 0, 0, MPI_COMM_WORLD);

		freeData(inputData, inputData.numOfSequences2 / 2);
	}

	MPI_Finalize();

	return 0;
}
