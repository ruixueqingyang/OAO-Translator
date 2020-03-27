CLANG_INSTALL = /home/ghn/work/llvm-project/install

CUDA_PATH = /opt/cuda

CUDA_INC = -I/usr/local/cuda/include

CLANG_INC = -I$(CLANG_INSTALL)/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

MY_INCLUDE = -I$(CLANG_INSTALL)/include -I./

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_35

OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

CUDA_SDK_PATH = /opt/cuda/samples
