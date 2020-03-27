#
SHELL = /bin/sh
.SUFFIXES: .c .o

include ../../path.mk

3mm_exec = 3mm

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	3mm.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $< $(DEBUG)

all: $(3mm_exec)

3mm: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: 3mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> 3mm_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: 3mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: 3mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> 3mm_full.ast
AST: 3mm.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > 3mm_full.ast

#cpu 单独
3mm_exec3 = CPU

SOURCES2.3 = \
	3mm_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(3mm_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
3mm_exec2 = OMP

SOURCES2.2 = \
	3mm_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(3mm_exec2)

# OMP: $(OBJECTS2.2)
# 	@echo "Linking"
# 	$(CXX) $(OBJECTS2.2) $(LDFLAGS) -o $@

OMP1: 3mm_omp.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: 3mm_omp.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: 3mm_omp.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO
OAOO: 3mm_out.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: 3mm_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin

#Dawncc_1
DAWN_1: 3mm_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2: 3mm_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN: 3mm_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin

#Dawncc_1
DAWN_1_native: 3mm_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2_native: 3mm_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN_native: 3mm_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
	
OAO_manual: 3mm_omp_out_manual.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_manual:
	./OAO_manual.bin
	
#Dawncc_modify1
DAWNModify1: 3mm_dawncc_1_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify2
DAWNModify2: 3mm_dawncc_2_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify
DAWNModify: 3mm_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin


clean:
	rm -f *.o *.bin OMP


