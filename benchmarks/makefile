# you can use this makefile to translate all bencchmarks

all:

polybench_all: syrk syr2k mvt gesummv gemm covar corr bicg atax fdtd2d 3mm 3dconv 2mm 2dconv

rodinia_al: cfd pre_cfd bfs srad lud lavaMD nw pf

OAO_DIR = ../src

# translate polybench

POLYBENCH_DIR = ./PolyBench

syrk:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/SYRK/syrk_omp.c > $(POLYBENCH_DIR)/plain/SYRK/OAOOut.txt
syr2k:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/SYR2K/syr2k_omp.c > $(POLYBENCH_DIR)/plain/SYR2K/OAOOut.txt
mvt:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/MVT/mvt_omp.c > $(POLYBENCH_DIR)/plain/MVT/OAOOut.txt
gesummv:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/GESUMMV/gesummv_omp.c > $(POLYBENCH_DIR)/plain/GESUMMV/OAOOut.txt
gemm:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/GEMM/gemm_omp.c > $(POLYBENCH_DIR)/plain/GEMM/OAOOut.txt
covar:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/COVAR/covariance_omp.c > $(POLYBENCH_DIR)/plain/COVAR/OAOOut.txt
corr:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/CORR/correlation_omp.c > $(POLYBENCH_DIR)/plain/CORR/OAOOut.txt
bicg:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/BICG/bicg_omp.c > $(POLYBENCH_DIR)/plain/BICG/OAOOut.txt
atax:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/ATAX/atax_omp.c > $(POLYBENCH_DIR)/plain/ATAX/OAOOut.txt
fdtd2d:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/FDTD-2D/fdtd2d_omp.c > $(POLYBENCH_DIR)/plain/FDTD-2D/OAOOut.txt
3mm:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/3MM/3mm_omp.c > $(POLYBENCH_DIR)/plain/3MM/OAOOut.txt
3dconv:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/3DCONV/3dconv_omp.c > $(POLYBENCH_DIR)/plain/3DCONV/OAOOut.txt
2mm:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/2MM/2mm_omp.c > $(POLYBENCH_DIR)/plain/2MM/OAOOut.txt
2dconv:
	$(OAO_DIR)/OAO.bin -fopenmp $(POLYBENCH_DIR)/plain/2DCONV/2dconv_omp.c > $(POLYBENCH_DIR)/plain/2DCONV/OAOOut.txt

	
# translate Rodinia

RODINIA_DIR = ./Rodina

cfd:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/cfd/euler3d_cpu.cpp > $(RODINIA_DIR)/openmp/cfd/OAOOut.txt
pre_cfd:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/cfd/pre_euler3d_cpu.cpp > $(RODINIA_DIR)/openmp/cfd/OAOOut_pre.txt
bfs:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/bfs/bfs.cpp > $(RODINIA_DIR)/openmp/bfs/OAOOut.txt
srad:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/srad/srad_v2/srad.cpp > $(RODINIA_DIR)/openmp/srad/srad_v2/OAOOut.txt
lud:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/lud/omp/lud_omp_one_file.c > $(RODINIA_DIR)/openmp/lud/omp/OAOOut.txt
lavaMD:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/lavaMD/OneFile.c > $(RODINIA_DIR)/openmp/lavaMD/OAOOut.txt
nw:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/nw/needle.cpp > $(RODINIA_DIR)/openmp/nw/OAOOut.txt
pf:
	$(OAO_DIR)/OAO.bin -fopenmp  $(RODINIA_DIR)/openmp/particlefilter/ex_particle_OPENMP_seq.c > $(RODINIA_DIR)/openmp/particlefilter/OAOOut.txt






