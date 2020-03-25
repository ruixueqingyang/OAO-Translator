INSTALL_PATH = ../../bin

include ../path.mk # include CLANG_INSTALL CUDA_INC

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_61

OFFLOADING_LDFLAGS = -I$(CUDA_PATH)/include -L$(CUDA_PATH)/nvvm/libdevice -L$(CUDA_PATH)/lib64 -L$(CLANG_INSTALL)/lib -lcudart

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

translate_demo:
	$(INSTALL_PATH)/OAO.bin -fopenmp ./demo.c > ./OAOOut.txt
	
RunTime.o: RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
demo_offloading: demo_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o  $@.bin
