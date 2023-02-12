all: assignment.cu
	nvcc assignment.cu accessory.cu -o assignment
