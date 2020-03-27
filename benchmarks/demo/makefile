INSTALL_PATH = ../../bin

include ../path.mk # include CLANG_INSTALL CUDA_INC

CFG: demo.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.DumpCFG -std=c++11 $<  2> demo_full.CFG # | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
VCFG: demo.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fno-color-diagnostics --analyze -Xanalyzer -analyzer-checker=debug.ViewCFG -std=c++11 $<
AST1: demo.c
	$(MY_LIB_PATH) $(MY_CLANG) -cc1 -fopenmp -fsyntax-only -ast-dump $< #> demo_full.ast
AST: demo.c
	$(MY_LIB_PATH) $(MY_CLANG) -fopenmp -fsyntax-only -Xclang -ast-dump $< -fno-color-diagnostics > demo_full.ast


translate_demo:
	$(INSTALL_PATH)/OAO.bin -fopenmp ./demo.c > ./OAOOut.txt
	
RunTime.o: RunTime.cpp RunTime.h
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $(EXPENSES) -c $< -o $@
demo_offloading: demo_out.c RunTime.h RunTime.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) -c $< -o  $@.o
	$(MY_CLANG) -O3 $(OFFLOADING_FLAGS) $@.o RunTime.o -o  $@.bin
