# OAO-Translator
A source-to-source translator that translates OpenMP CPU programs to OpenMP Offloading programs with data transmission optimization.  
You can compile OAO and run the tests by following these steps.

## Installation of OAO
### Compiler Preparation
1. Install GCC-8.3.0 and CUDA-10.1. (Higher version of CUDA is not supported by LLVM-9.0.0.)
1. Download source code of LLVM-9.0.0 release. (Other versions may not be compatible with our code, because of defferent API.)
1. Compile LLVM for the first time

    You can refer to the documentation of LLVM.  
    (https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
    
    Here are the key cmake parameters:  
    -DCMAKE_BUILD_TYPE=Release \  
    -DCMAKE_C_COMPILER=gcc \  
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \  
    -DLLVM_ENABLE_PROJECTS="llvm;clang;clang-tools-extra;compiler-rt;lld;openmp" \  
    -DLLVM_TOOL_CLANG_BUILD=ON \  
    -DLLVM_TOOL_COMPILER_RT_BUILD=ON \  
    -DLLVM_TOOL_LLD_BUILD=ON \  
    -DLLVM_TOOL_OPENMP_BUILD=ON \  

1. Compile LLVM for the second time to support OpenMP Offloading
    
    You can refer to the documentation of LLVM.  
    (https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
    
    Here are the key cmake parameters:  
    -DCUDA_HOST_COMPILER=gcc \  
    -DLIBOMPTARGET_NVPTX_CUDA_COMPILER=clang++ \ # the clang compiled in step 3  
    -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_xx \ # compute capability of NVIDIA GPU  
    -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=xx \ # Compute Capability of NVIDIA GPU  
    -DLIBOMPTARGET_NVPTX_ENABLE_BCLIB=TRUE \  
    -DCUDA_BUILD_CUBIN=ON \  

1. Patch some source files in LLVM installation path. (You can copy the patches to the corresponding path.)

    + LLVM_installation_path/include/clang/AST/RecursiveASTVisitor.h
    
        cd LLVM_installation_path/include/clang/AST/  
        Make a copy of RecursiveASTVisitor.h. (cp RecursiveASTVisitor.h RecursiveASTVisitorProtected.h)  
        Modify "private:" to "protected:" in RecursiveASTVisitorProtected.h. (Two places in total.)  
        
    + LLVM_installation_path/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h
    
        Comment line 49 to 52
        
    + LLVM_installation_path/lib/clang/9.0.0/include/__clang_cuda_cmath.h
    
        Comment line 55 to 58
    
1. Set the environment variables with PATH and LD_LIBRARY_PATH.  
    
### Compile and install OAO
1. Open the OAO-Translator/build directory.
2. Modify the CLANG_INSTALL and INSTALL_PATH in makefile.
3. make all && make install

## How to run the tests
### OAO parameter format  
        OAO_installation_path/OAO.bin -fopenmp <compilation_parameters> source_file

### Simple demo  
        cd OAO-Translator/benchmarks/demo && make translate_demo  
        make demo_offloading

### PolyBench and Rodinia
1. Modify pathes in OAO-Translator/benchmarks/path.mk  
1. Use OAO-Translator/benchmarks/makefile to translate benchmarks  
    cd OAO-Translator/benchmarks && make all # translate all benchmarks  
    cd OAO-Translator/benchmarks && make gemm # translate benchmark GEMM  
1. Use .sh files in OAO-Translator/benchmarks/shell_PolyBench/ and OAO-Translator/benchmarks/shell_Rodinia/ to run benchmarks and get performance data.
