###
### For COMS E6998 Spring 2024
### Instructor: Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

SDK_INSTALL_PATH := /usr/local/cuda-11.3
NVCC= $(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := pkh2120CUDA1.tar
EXECS :=  vecadd00 vecadd01 matmult00 matmult01
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
# This vecaddKernel00.cu program contains a kernel function that performs non-coalescing vector addition.
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
# This vecadd00 executable combines the listed files to execute a kernel function that performs non-coalescing vector addition and reports statistics about the nature of the program and kernel function execution.
vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


####################################################################### 
# This vecaddKernel01.cu program contains a kernel function that performs vector addition using coalescing memory accesses.
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
# This vecadd01 executable combines the listed files to execute a kernel function that performs vector addition with coalesing memory accesses and reports statistics about the nature of the program and kernel function execution.
vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Provided Kernel
# This matmultKernel00.cu program contains a kernel function that performs matrix multiplication where each thread computes just one value in the resultant output matrix.
matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)
# This matmult00 executable combines the listed files to execute a kernel function that performs matrix multiplication where each thread computes just one value in the resultant output matrix and reports statistics about the nature of the program and kernel function execution.
matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
# This matmultKernel01.cu program contains a kernel function that performs matrix multiplication where each thread computes four values in the resultant output matrix.
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32
###
# This matmult01 executable combines the listed files to execute a kernel function that performs matrix multiplication where each thread computes four values in the resultant output matrix and reports statistics about the nature of the program and kernel function execution.
matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32
