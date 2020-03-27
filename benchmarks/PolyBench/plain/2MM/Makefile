#
SHELL = /bin/sh
.SUFFIXES: .c .o

include ../../path.mk

2mm_exec = 2mm

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	2mm.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $< $(DEBUG)

all: $(2mm_exec)

2mm: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: 2mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> 2mm_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: 2mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: 2mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> 2mm_full.ast
AST: 2mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > 2mm_full.ast


#cpu 单独
2mm_exec3 = CPU

SOURCES2.3 = \
	2mm_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(2mm_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
2mm_exec2 = OMP

SOURCES2.2 = \
	2mm_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(2mm_exec2)

# OMP: $(OBJECTS2.2)
# 	@echo "Linking"
# 	$(CXX) $(OBJECTS2.2) $(LDFLAGS) -o $@

OMP1: 2mm_omp.c
	$(CXX) -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: 2mm_omp.c
	$(CXX) -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: 2mm_omp.c
	$(CXX) -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO
OAOO: 2mm_out.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: 2mm_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin

#Dawncc_1
DAWN_1: 2mm_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2: 2mm_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN: 2mm_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin

#Dawncc_1
DAWN_1_native: 2mm_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2_native: 2mm_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN_native: 2mm_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
	
OAO_manual: 2mm_omp_out_manual.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_manual:
	./OAO_manual.bin

#Dawncc_modify1
DAWNModify1: 2mm_dawncc_1_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify2
DAWNModify2: 2mm_dawncc_2_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify
DAWNModify: 2mm_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin


clean:
	rm -f *.o *.bin OMP


