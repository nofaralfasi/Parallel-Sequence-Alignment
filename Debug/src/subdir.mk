################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/calculations.c \
../src/main.c \
../src/programData.c 

O_SRCS += \
../src/calculations.o \
../src/cudaFunctions.o \
../src/main.o \
../src/programData.o 

OBJS += \
./src/calculations.o \
./src/main.o \
./src/programData.o 

C_DEPS += \
./src/calculations.d \
./src/main.d \
./src/programData.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	mpicc -fopenmp -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/main.o: ../src/main.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	mpicc -fopenmp -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"src/main.d" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


