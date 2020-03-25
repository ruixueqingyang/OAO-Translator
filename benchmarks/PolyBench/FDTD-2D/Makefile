#
SHELL = /bin/sh
.SUFFIXES: .c .o

CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = $(CLANG_INSTALL)/include

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

#OFFLOADING_LDFLAGS = -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -L/opt/cuda/targets/x86_64-linux/lib/stubs -L/opt/cuda/targets/x86_64-linux/lib -lomp -lomptarget -lm -lstdc++ -lcudart -lcudadevrt -lcudart_static
OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

fdtd2d_exec = fdtd2d

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	fdtd2d.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(fdtd2d_exec)

fdtd2d: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: fdtd2d.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> fdtd2d_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: fdtd2d.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: fdtd2d.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> fdtd2d_full.ast
AST: fdtd2d.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > fdtd2d_full.ast


#cpu 单独
fdtd2d_exec3 = CPU

SOURCES2.3 = \
	fdtd2d_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(fdtd2d_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
fdtd2d_exec2 = OMP

SOURCES2.2 = \
	fdtd2d_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(fdtd2d_exec2)

OMP1: fdtd2d_omp.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: fdtd2d_omp.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: fdtd2d_omp.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO
OAOO: fdtd2d_out.c
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: fdtd2d_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  -c $< -o $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  $@.o RunTime.o -o $@.bin

#Dawncc_1
DAWN_1: fdtd2d_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_2
DAWN_2: fdtd2d_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc
DAWN: fdtd2d_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

#Dawncc_1
DAWN_1_native: fdtd2d_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_2
DAWN_2_native: fdtd2d_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc
DAWN_native: fdtd2d_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin	

#cpu-kurnel
fdtd2d_exec4 = KERNEL

SOURCES2.4 = \
	fdtd2d_omp_kernel.c
OBJECTS2.4 = $(SOURCES2.4:.c=.o)

# OMPKERNEL: $(fdtd2d_exec4)

KERNEL: $(OBJECTS2.4)
	@echo "Linking"
	$(CXX) $(OBJECTS2.4) $(LDFLAGS) -o $@

OMPKERNEL1: fdtd2d_omp_kernel.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMPKERNEL2: fdtd2d_omp_kernel.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMPKERNEL: fdtd2d_omp_kernel.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO_kernel
OAOKERNEL: fdtd2d_omp_kernel_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  -c $< -o $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  $@.o RunTime.o -o $@.bin

#Dawncc_kernel_1
DAWNKERNEL1: fdtd2d_dawncc_kernel_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_kernel_2
DAWNKERNEL2: fdtd2d_dawncc_kernel_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_kernel
DAWNKERNEL: fdtd2d_dawncc_kernel.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

#Dawncc_kernel_1
DAWNKERNEL1_native: fdtd2d_dawncc_kernel_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_kernel_2
DAWNKERNEL2_native: fdtd2d_dawncc_kernel_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_kernel
DAWNKERNEL_native: fdtd2d_dawncc_kernel_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

OAO_manual: fdtd2d_omp_out_manual.c
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  -c $< -o $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  $@.o  -o $@.bin
run_manual:
	./OAO_manual.bin
	

#Dawncc_modify1
DAWNModify1: fdtd2d_dawncc_modify_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_modify2
DAWNModify2: fdtd2d_dawncc_modify_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_modify
DAWNModify: fdtd2d_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

run_DAWNModify:
	./DAWNModify.bin

clean:
	rm -f *.o *.bin OMP


