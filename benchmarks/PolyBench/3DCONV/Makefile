#
SHELL = /bin/sh
.SUFFIXES: .c .o

CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = -I$(CLANG_INSTALL)/include

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

3dconv_exec = 3dconv

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	3DConvolution.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $< $(DEBUG)

all: $(3dconv_exec)

3dconv: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: 3DConvolution.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> 3dconv_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: 3DConvolution.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: 3DConvolution.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> 3dconv_full.ast
AST: 3DConvolution.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > 3dconv_full.ast

#cpu 单独
3dconv_exec3 = CPU

SOURCES2.3 = \
	3DConvolution_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(3dconv_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
3dconv_exec2 = OMP

SOURCES2.2 = \
	3DConvolution_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(3dconv_exec2)

# OMP: $(OBJECTS2.2)
# 	@echo "Linking"
# 	$(CXX) $(OBJECTS2.2) $(LDFLAGS) -o $@

OMP1: 3DConvolution_omp.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: 3DConvolution_omp.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: 3DConvolution_omp.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO
OAOO: 3DConvolution_out.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: 3DConvolution_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin

#Dawncc_1
DAWN_1: 3DConvolution_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2: 3DConvolution_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN: 3DConvolution_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin

#Dawncc_1
DAWN_1_native: 3DConvolution_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2_native: 3DConvolution_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN_native: 3DConvolution_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
	
OAO_manual: 3DConvolution_omp_out_manual.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_manual:
	./OAO_manual.bin

#Dawncc_modify1
DAWNModify1: 3DConvolution_dawncc_1_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify2
DAWNModify2: 3DConvolution_dawncc_2_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify
DAWNModify: 3DConvolution_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin


clean:
	rm -f *.o *.bin OMP

