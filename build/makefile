# LLVM_INSTALL = /home/wfr/install/LLVM-9/install-9
LLVM_INSTALL = /home/ghn/work/llvm-project/install
LLVM_INSTALL := $(shell ./path.sh $(LLVM_INSTALL))
INSTALL_PATH = ../bin
LLVMCOMPONENTS := #cppbackend
RTTIFLAG := -fno-rtti
LLVMCONFIG := $(LLVM_INSTALL)/bin/llvm-config
CXXFLAGS := -I$(shell $(LLVMCONFIG) --src-root)/tools/clang/include -I$(shell $(LLVMCONFIG) --obj-root)/tools/clang/include -I/usr/include $(shell $(LLVMCONFIG) --cxxflags) $(RTTIFLAG) -L/usr/lib/x86_64-linux-gnu
LLVMLDFLAGS := $(shell $(LLVMCONFIG) --ldflags --libs $(LLVMCOMPONENTS))

CLANGLIBS = \
        -lclangTooling\
        -lclangFrontendTool\
        -lclangFrontend\
        -lclangDriver\
        -lclangSerialization\
        -lclangCodeGen\
        -lclangParse\
        -lclangSema\
        -lclangStaticAnalyzerFrontend\
        -lclangStaticAnalyzerCheckers\
        -lclangStaticAnalyzerCore\
        -lclangAnalysis\
        -lclangARCMigrate\
        -lclangRewrite\
        -lclangRewriteFrontend\
        -lclangEdit\
        -lclangAST\
        -lclangLex\
        -lclangBasic\
        $(shell $(LLVMCONFIG) --libs)\
        $(shell $(LLVMCONFIG) --system-libs)\
        -lncurses

MY_LIB_PATH = LIBRARY_PATH=$(LLVM_INSTALL)/lib:${LIBRARY_PATH}

MY_CLANG = $(LLVM_INSTALL)/bin/clang++

MY_INCLUDE = -I$(LLVM_INSTALL)/include -I../src

SRCS = ../src/BaseFuns.cpp ../src/FuncInfo.cpp ../src/OAOSubFuncs.cpp ../src/OAORewriter.cpp ../src/OAO.cpp

OBJS = $(SRCS:.cpp = .o)

TARGET = OAO

.PHONY: all clean

all: $(TARGET)

$(TARGET) : $(OBJS)
	$(MY_CLANG) $^ -o $@.bin $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE) -g -O0
%.o : %.cpp
	$(MY_CLANG) -c $< $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE) -g -O0
clean:
	rm -rf *.o *.bin
install:
	cp ./OAO.bin ../src/RunTime.cpp ../src/RunTime.h $(INSTALL_PATH)
demo:
	./OAO.bin -fopenmp ../benchmarks/demo/demo.c > ../benchmarks/demo/OAOOut.txt
