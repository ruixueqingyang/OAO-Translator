CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = -I$(CLANG_INSTALL)/include

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

CUDA_SDK_PATH = /opt/cuda/samples

# C compiler
CC = g++
ICC = icc
CC_FLAGS = -fopenmp -O3 # -g

all: bfs bfs_out

RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
bfs_out: bfs_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -O3 -DOFFLOADING $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o $@.bin
bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o $@.bin

run3:
	./bfs.bin 36 ../../data/bfs/graph4096.txt
run2:
	./bfs.bin 36 ../../data/bfs/graph65536.txt
run1:
	./bfs.bin 36 ../../data/bfs/graph1MW_6.txt

run_out3: 
	./bfs_out.bin 36 ../../data/bfs/graph4096.txt
run_out2: 
	./bfs_out.bin 36 ../../data/bfs/graph65536.txt
run_out1: 
	./bfs_out.bin 36 ../../data/bfs/graph1MW_6.txt

OAO_manual: bfs_out_manual.cpp
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_manual3:
	./OAO_manual.bin 36 ../../data/bfs/graph4096.txt
run_manual2:
	./OAO_manual.bin 36 ../../data/bfs/graph65536.txt
run_manual1:
	./OAO_manual.bin 36 ../../data/bfs/graph1MW_6.txt


SET_LD_LIBRARY_PATH:
	LD_LIBRARY_PATH=$(CLANG_INSTALL)/lib:$(LD_LIBRARY_PATH)

# bfs_offload: bfs.cpp
# 	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f *.o result_out.txt
	rm -rf *.o
	rm -rf *.bin
	rm -rf *.out
	rm -rf bfs
