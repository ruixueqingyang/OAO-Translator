#makefile

CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = -I$(CLANG_INSTALL)/include

CUDA_INC = -I/opt/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)


openmp: ex_particle_OPENMP_seq.c
	gcc -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm 

ONE: ex_particle_OPENMP_seq.c
	$(MY_CLANG) -O3 -ffast-math -lm -fopenmp $< -o $@.bin
run_ONE:
	./ONE.bin -x 128 -y 128 -z 10 -np 10000

RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
OUT: ex_particle_OPENMP_seq_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 -ffast-math -lm $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O3 -ffast-math -lm $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin
CPP: ex_particle_OPENMP_seq_out.cpp RunTime.h RunTime.o
	$(MY_CLANG) -g -O0 -ffast-math -lm $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -g -O0 -ffast-math -lm $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin

OAO_manual: ex_particle_OPENMP_seq_out_manual.c RunTime.h RunTime.o
	$(MY_CLANG) -O0 -ffast-math -lm $(OFFLOADING_FLAGS) $(DEBUG) -c $< -o $@.o
	$(MY_CLANG) -O0 -ffast-math -lm $(OFFLOADING_FLAGS) $(DEBUG) $@.o RunTime.o -o $@.bin

run_OUT:
	./OUT.bin -x 128 -y 128 -z 10 -np 10000
run_CPP:
	./CPP.bin -x 128 -y 128 -z 10 -np 10000
run_manual:
	./OAO_manual.bin -x 128 -y 128 -z 10 -np 10000

icc_openmp: ex_particle_OPENMP_seq.c
	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter


clean:
	rm -rf *.o
	rm -rf *.bin