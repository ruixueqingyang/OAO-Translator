# Makefile for LU Decomposition
#
#
CLANG_INSTALL = /home/wfr/install/LLVM-9/install-9

CLANG_INC = -I$(CLANG_INSTALL)/include

CUDA_INC = -I/usr/local/cuda/include

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:/opt/cuda/lib64:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

OFFLOADING_TARGET_FLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75

OFFLOADING_LDFLAGS = $(CUDA_INC) -L/opt/cuda/nvvm/libdevice -L/opt/cuda/lib64 -L$(CLANG_INSTALL)/lib -lcudart #-lomptarget

OFFLOADING_FLAGS = $(OFFLOADING_TARGET_FLAGS) $(OFFLOADING_LDFLAGS)

CUDA_SDK_PATH = /opt/cuda/samples


# DEBUG can be set to YES to include debugging info, or NO otherwise
DEBUG          := NO

# PROFILE can be set to YES to include profiling info, or NO otherwise
PROFILE        := NO

# ------------  name of the executable  ----------------------------------------
EXECUTABLE      := lud_omp

# ------------  list of all source files  --------------------------------------
SOURCES         := lud.c lud_omp.c ../common/common.c 

# ------------  compiler  ------------------------------------------------------
CC              := gcc
CXX             := g++

# ------------  compiler flags  ------------------------------------------------
DEBUG_CFLAGS    := -Wall -O0 #-g
RELEASE_CFLAGS  := -Wall -O3 
COMMON_CFLAGS   := -fopenmp

# ------------  compiler defs  ------------------------------------------------
DEBUG_CDEFS    := 
RELEASE_CDEFS  := 
COMMON_CDEFS   := 

# ------------  linker flags  --------------------------------------------------
DEBUG_LDFLAGS    := #-g
RELEASE_LDFLAGS  := 
COMMON_LDFLAGS   := -fopenmp 

ifeq (YES, ${DEBUG})
  CFLAGS       := ${DEBUG_CFLAGS} ${DEBUG_CDEFS}
  CXXFLAGS     := ${DEBUG_CXXFLAGS} ${DEBUG_CDEFS}
  LDFLAGS      := ${DEBUG_LDFLAGS}
else
  CFLAGS       := ${RELEASE_CFLAGS} ${RELEASE_CDEFS}
  CXXFLAGS     := ${RELEASE_CXXFLAGS} ${RELEASE_CDEFS}
  LDFLAGS      := ${RELEASE_LDFLAGS}
endif

CFLAGS         := ${CFLAGS} ${COMMON_CFLAGS}
CXXFLAGS       := ${CXXFLAGS} ${COMMON_CFLAGS}
LDFLAGS        := ${LDFLAGS} ${COMMON_LDFLAGS}

ifeq (YES, ${PROFILE})
  CFLAGS       := ${CFLAGS}   -pg -O3
  CXXFLAGS     := ${CXXFLAGS} -pg -O3
  LDFLAGS      := ${LDFLAGS}  -pg
endif

# ------------  additional system include directories  -------------------------
GLOBAL_INC_DIR  =

# ------------  private include directories  -----------------------------------
LOCAL_INC_DIR   = ../common

# ------------  system libraries  (e.g. -lm )  ---------------------------------
SYS_LIBS        = -lm

# ------------  additional system library directories  -------------------------
GLOBAL_LIB_DIR  =

# ------------  additional system libraries  -----------------------------------
GLOBAL_LIBS     =

# ------------  private library directories  -----------------------------------
LOCAL_LIB_DIR   = 

# ------------  private libraries  (e.g. libxyz.a )  ---------------------------
LOCAL_LIBS      =


C_SOURCES       = $(filter     %.c, $(SOURCES))
CPP_SOURCES     = $(filter-out %.c, $(SOURCES))
ALL_INC_DIR     = $(addprefix -I, $(LOCAL_INC_DIR) $(GLOBAL_INC_DIR))
ALL_LIB_DIR     = $(addprefix -L, $(LOCAL_LIB_DIR) $(GLOBAL_LIB_DIR))
GLOBAL_LIBSS    = $(addprefix $(GLOBAL_LIB_DIR)/, $(GLOBAL_LIBS))
LOCAL_LIBSS     = $(addprefix $(LOCAL_LIB_DIR)/, $(LOCAL_LIBS))
ALL_CFLAGS      = $(CFLAGS) $(ALL_INC_DIR)
ALL_LFLAGS      = $(LDFLAGS) $(ALL_LIB_DIR)
BASENAMES       = $(basename $(SOURCES))

# ------------  generate the names of the object files  ------------------------
OBJECTS         = $(addsuffix .o,$(BASENAMES))
 
# ------------  make the executable (the default goal)  ------------------------
$(EXECUTABLE):	$(OBJECTS)
ifeq ($(strip $(CPP_SOURCES)),)
	$(CC)  $(ALL_LFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LOCAL_LIBSS) $(GLOBAL_LIBSS) $(SYS_LIBS)
else
	$(CXX) $(ALL_LFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LOCAL_LIBSS) $(GLOBAL_LIBSS) $(SYS_LIBS)
endif

one: 
	$(CXX) $(ALL_LFLAGS) lud_omp_one_file.c -o $@.bin $(LOCAL_LIBSS) $(GLOBAL_LIBSS) $(SYS_LIBS)
out: RunTime.o lud_omp_one_file_out.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(SYS_LIBS) $^ -o $@.bin
RunTime.o:RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) -c $< -o $@
lud_omp_one_file_out.o: lud_omp_one_file_out.c 
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) lud_omp_one_file_out.c -c
run_one:
	./one.bin -s 8000 
run_out:
	./out.bin -s 8000 

OAO_manual: lud_omp_one_file_out_manual.c
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o  -o  $@.bin
run_out_manual:
	./OAO_manual.bin -s 8000


# ------------  make the objects  ----------------------------------------------
%.o:		%.c
				$(CC)  -c $(ALL_CFLAGS) $< -o $@

%.o:		%.cc
				$(CXX) -c $(ALL_CFLAGS) $< -o $@

%.o:		%.cpp
				$(CXX) -c $(ALL_CFLAGS) $< -o $@

%.o:		%.C
				$(CXX) -c $(ALL_CFLAGS) $< -o $@

clean:
	-rm --force $(EXECUTABLE) $(OBJECTS) *~
	rm *.o *.bin

.PHONY: clean

# ==============================================================================
# vim: set tabstop=2: set shiftwidth=2:
