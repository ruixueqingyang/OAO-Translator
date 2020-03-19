#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make bfs

cd /home/anjushi/work/par/Rodinia_3.1/openmp/bfs

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/runtime_3.txt"
FILE5="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/profile_3_oao.txt"
FILE6="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/profile_3_manual.txt"

FILE11="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/runtime_2.txt"
FILE55="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/profile_2_oao.txt"
FILE66="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/profile_2_manual.txt"

FILE111="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/runtime_1.txt"
FILE555="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/profile_1_oao.txt"
FILE666="/home/anjushi/work/par/Rodinia_3.1/openmp/bfs/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6
rm -rf $FILE11  $FILE55  $FILE66
rm -rf $FILE111  $FILE555  $FILE666

make clean

#make bfs  #OMP
make EXPENSES=-DEXPENSES bfs_out   #OAO
make OAO_manual   #manual

for ((i=1; i<=$times; i ++))
do  

   `nvprof --csv --print-gpu-trace ./bfs_out.bin 4 ../../data/bfs/graph4096.txt 1>> runtime_3.txt 2>> profile_3_oao.txt`  #OAO
   `nvprof --csv --print-gpu-trace ./bfs_out.bin 4 ../../data/bfs/graph65536.txt 1>> runtime_2.txt 2>> profile_2_oao.txt`  #OAO
   `nvprof --csv --print-gpu-trace ./bfs_out.bin 4 ../../data/bfs/graph1MW_6.txt 1>> runtime_1.txt 2>> profile_1_oao.txt`  #OAO

   `nvprof --csv --print-gpu-trace ./OAO_manual.bin 4 ../../data/bfs/graph4096.txt 1>> runtime_3.txt 2>> profile_3_manual.txt`  #manual
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin 4 ../../data/bfs/graph65536.txt 1>> runtime_2.txt 2>> profile_2_manual.txt`  #manual
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin 4 ../../data/bfs/graph1MW_6.txt 1>> runtime_1.txt 2>> profile_1_manual.txt`  #manual

done
