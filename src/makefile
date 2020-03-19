CLANG_INSTALL = /home/ghn/work/llvm-project/install
CXX := $(CLANG_INSTALL)/bin/clang++
LLVMCOMPONENTS := #cppbackend
RTTIFLAG := -fno-rtti
LLVMCONFIG := $(CLANG_INSTALL)/bin/llvm-config
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


#$(CXX) -o $@ $< $(CLANGLIBS) $(LLVMLDFLAGS)
GCC8_INC = -I/home/wfr/install/GCC-8/install/include/c++/8.3.0 -I/home/wfr/install/GCC-8/install/include/c++/8.3.0/x86_64-pc-linux-gnu #-I/home/wfr/install/GCC-8/install/include/c++/8.3.0/tr1

MY_LIB_PATH = LIBRARY_PATH=$(CLANG_INSTALL)/lib:${LIBRARY_PATH}

MY_CLANG = $(CLANG_INSTALL)/bin/clang++

MY_INCLUDE = -I$(CLANG_INSTALL)/include -I./

CHESTNUT = /home/wfr/work/Coding/chestnut
CODING = /home/wfr/work/Coding

# 正则表达式表示目录下所有.c文件, 相当于：SRCS = main.c a.c b.c
# SRCS = $(wildcard *.c)
SRCS = BaseFuns.cpp FuncInfo.cpp OAOSubFuncs.cpp OAORewriter.cpp OAO.cpp

# OBJS表示SRCS中把列表中的.cpp全部替换为.o, 相当于：OBJS = main.o a.o b.o
# OBJS = $(patsubst %c, %o, $(SRCS))
OBJS = $(SRCS:.cpp = .o)    #OBJS将$(SRCS)下的.cpp文件转化为.o文件

# 目标名称
TARGET = OAO

# .PHONE伪目标
.PHONY: all clean run

all: $(TARGET)


# 第一行依赖关系：冒号后面为依赖的文件, 相当于Hello: main.o a.o b.o
# 第二行规则：$@表示目标文件, $^表示所有依赖文件, $<表示第一个依赖文件
$(TARGET) : $(OBJS)
	$(CXX) $^ -o $@.bin $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE) -Wall -g -O0

# 上一句目标文件依赖一大堆.o文件, 这句表示所有.o都由相应名字的.cpp文件自动生成
%.o : %.cpp
	$(CXX) -c $< $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE) -Wall -g -O0

OAORewriter: OAORewriter.cpp
	$(CXX) -c $< -o $@.o $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE)
OAOMain: OAO.cpp
	$(CXX) -c $< -o $@.o $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE)
BIN: OAORewriter.o OAOMain.o
	$(CXX) $^ -o $@.bin $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS) $(MY_INCLUDE)

CIrewriter: CIrewriter.cpp
	$(CXX) -o $@.bin $^ $(CXXFLAGS) $(LLVMLDFLAGS) $(CLANGLIBS)  $(MY_INCLUDE)
#	$(MY_LIB_PATH) g++ -I $(MY_INCLUDE) -O3 -std=c++11 $< -o $@.bin

clean:
	rm -rf *.bin *.o
	
run:
	./OAO.bin -fopenmp /home/wfr/work/Coding/chestnut/scenario_OMP.cpp > $(CHESTNUT)/OAOOut.txt
gdb:
	gdb ./OAO.bin
CORR:
	./OAO.bin -fopenmp /home/wfr/work/Coding/chestnut/correlation.c > $(CHESTNUT)/OAOOut.txt
gemm:
	./OAO.bin -fopenmp /home/wfr/work/Coding/chestnut/gemm.c > $(CHESTNUT)/OAOOut.txt
omp_base:
	./OAO.bin -fopenmp -I/usr/include/c++/9.1.0 -I/usr/include/c++/9.1.0/x86_64-pc-linux-gnu $(CHESTNUT)/omp_base.cc > $(CHESTNUT)/OAOOut.txt
fdtd2d:
	./OAO.bin -fopenmp /home/wfr/work/Coding/chestnut/fdtd2d.c > $(CHESTNUT)/OAOOut.txt
correlation:
	./OAO.bin -fopenmp /home/wfr/work/Coding/chestnut/correlation.c > $(CHESTNUT)/OAOOut.txt
one:
	./OAO.bin -v -g -O3 -fopenmp $(GCC8_INC) -Wall -lm /home/wfr/work/LULESH/OneCpp/lulesh.cc > /home/wfr/work/LULESH/OneCpp/OAOOut.txt
one1:
	./OAO.bin -fopenmp -lm /home/wfr/work/LULESH/OneCpp/lulesh.cc > /home/wfr/work/LULESH/OneCpp/OAOOut.txt
cfd:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Rodinia_3.1/openmp/cfd/euler3d_cpu.cpp > /home/wfr/work/Coding/Rodinia_3.1/openmp/cfd/OAOOut.txt
pre_cfd:
	./OAO.bin -fopenmp $(CODING)/Rodinia_3.1/openmp/cfd/pre_euler3d_cpu.cpp > $(CHESTNUT)/OAOOut.txt
pre_cfd_double:
	./OAO.bin -fopenmp $(CODING)/Rodinia_3.1/openmp/cfd/pre_euler3d_cpu_double.cpp > $(CHESTNUT)/OAOOut.txt
bfs:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Rodinia_3.1/openmp/bfs/bfs.cpp > $(CODING)/Rodinia_3.1/openmp/bfs/OAOOut.txt
kmeans:
	./OAO.bin -fopenmp $(CODING)/Rodinia_3.1/openmp/kmeans/kmeans_openmp/kmeans_one_file.c > $(CHESTNUT)/OAOOut.txt
lud:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Rodinia_3.1/openmp/lud/omp/lud_omp_one_file.c > /home/wfr/work/Coding/Rodinia_3.1/openmp/lud/omp/OAOOut.txt
lavaMD:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Rodinia_3.1/openmp/lavaMD/OneFile.c > $(CODING)/Rodinia_3.1/openmp/lavaMD/OAOOut.txt
nw:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Rodinia_3.1/openmp/nw/needle.cpp > $(CODING)/Rodinia_3.1/openmp/nw/OAOOut.txt
pf:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Rodinia_3.1/openmp/particlefilter/ex_particle_OPENMP_seq.c > $(CODING)/Rodinia_3.1/openmp/particlefilter/OAOOut.txt

bicg:
	./OAO.bin -fopenmp /home/wfr/work/Coding/Polybench/BICG/bicg.c > /home/wfr/work/Coding/Polybench/BICG/OAOOut.txt
atax:
	./OAO.bin -fopenmp /home/wfr/work/Polybench/atax.c > /home/wfr/work/Polybench/OAOOut.txt
syr2k:
	./OAO.bin -fopenmp /home/wfr/work/Polybench/syr2k.c > /home/wfr/work/Polybench/OAOOut.txt
	
2dconv:
	./OAO.bin -fopenmp -I/home/ghn/gcc_builder_7.3/build/gcc/include -I//home/ghn/gcc_builder_7.3/build/prev-x86_64-linux-gnu/libgomp /home/ghn/桌面/plain-20200207/2DCONV/2DConvolution_omp.c > /home/ghn/桌面/plain-20200207/2DCONV/OAOOut.txt

corr:
	./OAO.bin -fopenmp -I/home/ghn/gcc_builder_7.3/build/gcc/include -I//home/ghn/gcc_builder_7.3/build/prev-x86_64-linux-gnu/libgomp /home/ghn/桌面/plain-20200207/CORR/correlation_omp.c > /home/ghn/桌面/plain-20200207/CORR/OAOOut.txt
	
demo:
	./OAO.bin -fopenmp -I/home/ghn/gcc_builder_7.3/build/gcc/include -I//home/ghn/gcc_builder_7.3/build/prev-x86_64-linux-gnu/libgomp /home/ghn/桌面/demo/demo.c > /home/ghn/桌面/demo/OAOOut.txt

# -DUSE_MPI=0 -v -g -O3 -fopenmp -I. -Wall -lm /home/wfr/work/LULESH/OneCpp/lulesh.cc
# $@ 目标文件
# $^ 所有的依赖文件
# $< 第一个依赖文件
# -fopenmp -I/home/wfr/install/GCC-8/install/include/c++/8.3.0 -I/home/wfr/install/GCC-8/install/include/c++/8.3.0/x86_64-pc-linux-gnu /home/wfr/work/Rodinia_3.1/openmp/cfd/euler3d_cpu.cpp
