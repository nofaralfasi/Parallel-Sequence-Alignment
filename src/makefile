build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c programData.c -o programData.o
	mpicxx -fopenmp -c calculations.c -o calculations.o
	mpicxx -fopenmp -c ompFunctions.c -o ompFunctions.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o ParallelSequenceAlignment main.o programData.o calculations.o ompFunctions.o cudaFunctions.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./ParallelSequenceAlignment

run:
	mpiexec -np 2 ./ParallelSequenceAlignment

runOn2:
	mpiexec -np 2 -machinefile mf -map-by node ./ParallelSequenceAlignment

