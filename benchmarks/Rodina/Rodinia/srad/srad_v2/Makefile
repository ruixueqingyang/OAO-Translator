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
CC_FLAGS = -fopenmp -O3

srad: 
	$(CC) $(CC_FLAGS) srad.cpp -o $@.bin

RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
srad_out: srad_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -O3 -DOFFLOADING $(OFFLOADING_FLAGS) -c $< -o $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o $@.bin


run0:
	./srad 1024 1024 0 127 0 127 24 0.5 8
run_out0:
	./srad_out.bin 1024 1024 0 127 0 127 24 0.5 8

OAO_manual: srad_out_manual.cpp
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(DEBUG) $@.o  -o  $@.bin
run_out_manual0:
	./OAO_manual.bin 1024 1024 0 127 0 127 24 0.5 8

clean:
	rm *.o *.bin

# 数据量: 1024 1024
# 线程数: 24
# 迭代次数: 8