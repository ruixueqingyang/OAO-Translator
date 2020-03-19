#
SHELL = /bin/sh
.SUFFIXES: .c .o

CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = -I$(CLANG_INSTALL)/include

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

#OFFLOADING_LDFLAGS = -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -L/opt/cuda/targets/x86_64-linux/lib/stubs -L/opt/cuda/targets/x86_64-linux/lib -lomp -lomptarget -lm -lstdc++ -lcudart -lcudadevrt -lcudart_static
OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)


syrk_exec = syrk

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	syrk.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<  

all: $(syrk_exec)

syrk: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: syrk.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> syrk_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: syrk.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: syrk.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> syrk_full.ast
AST: syrk.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > syrk_full.ast


#cpu 单独
syrk_exec3 = CPU

SOURCES2.3 = \
	syrk_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(syrk_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
syrk_exec2 = OMP

SOURCES2.2 = \
	syrk_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(syrk_exec2)

# OMP: $(OBJECTS2.2)
# 	@echo "Linking"
# 	$(CXX) $(OBJECTS2.2) $(LDFLAGS) -o $@
OMP1: syrk_omp.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: syrk_omp.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: syrk_omp.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin
#OAO
OAOO: syrk_out.cpp
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: syrk_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) $@.o RunTime.o -o  $@.bin

#Dawncc
DAWN_1: syrk_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o  $@.bin
#Dawncc
DAWN_2: syrk_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o  $@.bin
#Dawncc
DAWN: syrk_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o  $@.bin

#Dawncc
DAWN_1_native: syrk_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o  $@.bin
#Dawncc
DAWN_2_native: syrk_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o  $@.bin
#Dawncc
DAWN_native: syrk_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o  $@.bin

run:
	./OMP 
run_dawn: SET_LD_LIBRARY_PATH
	./DAWN_1.bin
run_oao: SET_LD_LIBRARY_PATH
	./OAO.bin

SET_LD_LIBRARY_PATH:
	LD_LIBRARY_PATH=$(CLANG_INSTALL)/lib:$(LD_LIBRARY_PATH)

OAO_manual: syrk_omp_out_manual.c
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) $@.o  -o  $@.bin
run_manual:
	./OAO_manual.bin


#Dawncc_modify1
DAWNModify1: syrk_dawncc_1_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin
#Dawncc_modify2
DAWNModify2: syrk_dawncc_2_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin
#Dawncc_modify
DAWNModify: syrk_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin



clean:
	/bin/rm -f *.o *.bin OMP


