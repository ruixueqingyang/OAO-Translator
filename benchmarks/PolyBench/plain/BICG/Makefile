#
SHELL = /bin/sh
.SUFFIXES: .c .o

include ../../path.mk

bicg_exec = bicg

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	bicg.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $< $(DEBUG)

all: $(bicg_exec)

bicg: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: bicg.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> bicg_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: bicg.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: bicg.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> bicg_full.ast
AST: bicg.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > bicg_full.ast


#cpu 单独
bicg_exec3 = CPU

SOURCES2.3 = \
	bicg_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(bicg_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
bicg_exec2 = OMP

SOURCES2.2 = \
	bicg_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(bicg_exec2)

# OMP: $(OBJECTS2.2)
# 	@echo "Linking"
# 	$(CXX) $(OBJECTS2.2) $(LDFLAGS) -o $@

OMP1: bicg_omp.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: bicg_omp.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: bicg_omp.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO
OAOO: bicg_out.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: bicg_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin

#Dawncc_1
DAWN_1: bicg_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2: bicg_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN: bicg_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
	
#Dawncc_1
DAWN_1_native: bicg_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_2
DAWN_2_native: bicg_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc
DAWN_native: bicg_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin

OAO_manual: bicg_omp_out_manual.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_manual:
	./OAO_manual.bin

#Dawncc_modify1
DAWNModify1: bicg_dawncc_1_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify2
DAWNModify2: bicg_dawncc_2_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin
#Dawncc_modify
DAWNModify: bicg_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $< -o $@.bin


clean:
	rm -f *.o *.bin OMP
