#
SHELL = /bin/sh
.SUFFIXES: .c .o

include ../../path.mk

covar_exec = covar

SERCXX = gcc
CXX = $(SERCXX)

SOURCES2.0 = \
	covariance.c
OBJECTS2.0 = $(SOURCES2.0:.c=.o)

#Default build suggestions with OpenMP for g++
CXXFLAGS = -O3 -fopenmp
LDFLAGS = -O3 -fopenmp

.c.o:
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $< 

all: $(covar_exec)

covar: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -o $@

CFG: covariance.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> covar_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: covariance.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: covariance.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> covar_full.ast
AST: covariance.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > covar_full.ast


#cpu 单独
covar_exec3 = CPU

SOURCES2.3 = \
	covariance_cpu.c
OBJECTS2.3 = $(SOURCES2.3:.c=.o)

CPU0: $(covar_exec3)

CPU: $(OBJECTS2.3)
	@echo "Linking"
	$(CXX) $(OBJECTS2.3) $(LDFLAGS) -o $@

#cpu-omp
covar_exec2 = OMP

SOURCES2.2 = \
	covariance_omp.c
OBJECTS2.2 = $(SOURCES2.2:.c=.o)

OMP0: $(covar_exec2)

# OMP: $(OBJECTS2.2)
# 	@echo "Linking"
# 	$(CXX) $(OBJECTS2.2) $(LDFLAGS) -o $@

OMP1: covariance_omp.c
	gcc -O3 -fopenmp -D_DEBUG_1 $< -o $@.bin
OMP2: covariance_omp.c
	gcc -O3 -fopenmp -D_DEBUG_2 $< -o $@.bin
OMP: covariance_omp.c
	gcc -O3 -fopenmp $(DEBUG) $< -o $@.bin

#OAO
OAOO: covariance_out.c
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS) $< -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OAO: covariance_omp_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  -c $< -o $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  $@.o RunTime.o -o $@.bin


#Dawncc_1
DAWN_1: covariance_dawncc_1.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_2
DAWN_2: covariance_dawncc_2.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc
DAWN: covariance_dawncc.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

#Dawncc_1
DAWN_1_native: covariance_dawncc_1_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_2
DAWN_2_native: covariance_dawncc_2_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc
DAWN_native: covariance_dawncc_native.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

#cpu-kurnel
covar_exec4 = KERNEL

SOURCES2.4 = \
	covariance_kernel.c
OBJECTS2.4 = $(SOURCES2.4:.c=.o)

OMPKERNEL: $(covar_exec4)

KERNEL: $(OBJECTS2.4)
	@echo "Linking"
	$(CXX) $(OBJECTS2.4) $(LDFLAGS) -o $@

#Dawncc_kernel
DAWNKERNEL: covariance_dawncc_kernel.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin

#OAO_kernel
OAOKERNEL: covariance_kernel_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  -c $< -o $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  $@.o RunTime.o -o $@.bin

OAO_manual: covariance_omp_out_manual.c
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  -c $< -o  $@.o
	$(MY_CLANG) -O3 $(DEBUG) $(OFFLOADING_FLAGS)  $@.o  -o  $@.bin
run_manual:
	./OAO_manual.bin

#Dawncc_modify1
DAWNModify1: covariance_dawncc_1_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_modify2
DAWNModify2: covariance_dawncc_2_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin
#Dawncc_modify
DAWNModify: covariance_dawncc_modify.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS)  $< -o $@.bin


clean:
	rm -f *.o *.bin OMP


