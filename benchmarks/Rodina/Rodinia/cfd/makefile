CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = -I$(CLANG_INSTALL)/include

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

CUDA_SDK_PATH = /opt/cuda/samples

OMP_NUM_THREADS = 36
MIC_OMP_NUM_THREADS = 216
GPU_THREADS = 2
GPU_THREADS_PRE = 2
# OFFLOADING_PARM0 = 2
# OFFLOADING_PARM1 = 1

# RunTime0.o:RunTime0.cpp RunTime0.h
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
# pre_euler3d_cpu_out0: pre_euler3d_cpu_out0.cpp RunTime0.h RunTime0.o
# 	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS) $(OFFLOADING_FLAGS) -c $< -o $@.o
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime0.o -o $@.bin

# RunTime1.o:RunTime1.cpp RunTime1.h
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
# pre_euler3d_cpu_out1: pre_euler3d_cpu_out0.cpp RunTime1.h RunTime1.o
# 	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS) $(OFFLOADING_FLAGS) -c $< -o $@.o
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime1.o -o $@.bin

# pre_euler3d_cpu_out2: pre_euler3d_cpu_out1.cpp RunTime0.h RunTime0.o
# 	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS) $(OFFLOADING_FLAGS) -c $< -o $@.o
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime0.o -o $@.bin
# -g -O0
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
euler3d_cpu_out: euler3d_cpu_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS) $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o $@.bin
pre_euler3d_cpu_out: pre_euler3d_cpu_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS_PRE) $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o $@.bin
pre_euler3d_cpu_double_out: pre_euler3d_cpu_double_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS_PRE) $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o $@.bin
CFG: pre_euler3d_cpu.cpp
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> $<.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: pre_euler3d_cpu.cpp
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST: euler3d_cpu.cpp
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > $<.AST

euler3d_cpu_test: euler3d_cpu_test.cpp
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS) $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o -o $@.bin

pre_euler3d_cpu_test: pre_euler3d_cpu_test.cpp
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS_PRE) $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o -o $@.bin


run_all: run run_out run_pre run_pre_out

run3:
	./euler3d_cpu ../../data/cfd/fvcorr.domn.097K
run2:
	./euler3d_cpu ../../data/cfd/fvcorr.domn.193K
run1:
	./euler3d_cpu ../../data/cfd/missile.domn.0.2M

run_out3:
	./euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.097K
run_out2:
	./euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.193K
run_out1:
	./euler3d_cpu_out.bin ../../data/cfd/missile.domn.0.2M


run_pre3:
	./pre_euler3d_cpu ../../data/cfd/fvcorr.domn.097K
run_pre2:
	./pre_euler3d_cpu ../../data/cfd/fvcorr.domn.193K
run_pre1:
	./pre_euler3d_cpu ../../data/cfd/missile.domn.0.2M

run_pre_out3:
	./pre_euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.097K
run_pre_out2:
	./pre_euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.193K
run_pre_out1:
	./pre_euler3d_cpu_out.bin ../../data/missile.domn.0.2M


run_test:
	./euler3d_cpu_test.bin ../../data/cfd/fvcorr.domn.193K
run_pre_test:
	./pre_euler3d_cpu_test.bin ../../data/cfd/fvcorr.domn.193K
	

all: euler3d_cpu euler3d_cpu_out pre_euler3d_cpu pre_euler3d_cpu_out

#euler3d: euler3d.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 euler3d.cu -o euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

#euler3d_double: euler3d_double.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 euler3d_double.cu -o euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

euler3d_cpu: euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu.cpp -o $@.bin

# euler3d_cpu_offload:
# 	icc -g -O3 -DOMP_OFFLOAD -Dblock_length=$(MIC_OMP_NUM_THREADS) -fopenmp -offload-option,mic,compiler,"-no-opt-prefetch" euler3d_cpu.cpp -o euler3d_cpu_offload

euler3d_cpu_double: euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu_double.cpp -o $@.bin

#pre_euler3d: pre_euler3d.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 pre_euler3d.cu -o pre_euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

#pre_euler3d_double: pre_euler3d_double.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 pre_euler3d_double.cu -o pre_euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib  -lcutil

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu.cpp -o $@.bin

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu_double.cpp -o $@.bin


# OAO_manual: euler3d_cpu_out_manual.cpp
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
OAO_manual: euler3d_cpu_out_manual.cpp
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS) $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o -o $@.bin
	
run_manual3:
	./OAO_manual.bin ../../data/cfd/fvcorr.domn.097K
run_manual2:
	./OAO_manual.bin ../../data/cfd/fvcorr.domn.193K
run_manual1:
	./OAO_manual.bin ../../data/cfd/missile.domn.0.2M


OAO_manual_pre: pre_euler3d_cpu_out_manual.cpp
	$(MY_CLANG) -O3 -DOFFLOADING -Dblock_length=$(GPU_THREADS_PRE) $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o -o $@.bin

# OAO_manual_pre: pre_euler3d_cpu_out_manual.cpp
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
# 	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_manual_pre3:
	./OAO_manual_pre.bin ../../data/cfd/fvcorr.domn.097K
run_manual_pre2:
	./OAO_manual_pre.bin ../../data/cfd/fvcorr.domn.193K
run_manual_pre1:
	./OAO_manual_pre.bin ../../data/cfd/missile.domn.0.2M


clean:
	rm *.o *.bin
	rm -f euler3d
	rm -f euler3d_cpu
	rm -f euler3d_cpu_offload
	rm -f euler3d_double
	rm -f euler3d_cpu_double

	rm -f pre_euler3d
	rm -f pre_euler3d_cpu
	rm -f pre_euler3d_double
	rm -f pre_euler3d_cpu_double
