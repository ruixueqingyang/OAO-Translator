#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make cfd

cd /home/anjushi/work/par/Rodinia_3.1/openmp/cfd

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/runtime_6.txt"
FILE5="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_6_oao.txt"
FILE6="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_6_manual.txt"

FILE11="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/runtime_5.txt"
FILE55="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_5_oao.txt"
FILE66="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_5_manual.txt"

FILE111="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/runtime_4.txt"
FILE555="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_4_oao.txt"
FILE666="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_4_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6
rm -rf $FILE11  $FILE55  $FILE66
rm -rf $FILE111  $FILE555  $FILE666

make clean

make pre_euler3d_cpu  #OMP
make EXPENSES=-DEXPENSES pre_euler3d_cpu_out   #OAO
make OAO_manual_pre   #manual

for ((i=1; i<=$times; i ++))
do  

   `nvprof --csv --print-gpu-trace ./pre_euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.097K 1>> runtime_6.txt 2>> profile_6_oao.txt`  #OAO
   `nvprof --csv --print-gpu-trace ./pre_euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.193K 1>> runtime_5.txt 2>> profile_5_oao.txt`  #OAO
   `nvprof --csv --print-gpu-trace ./pre_euler3d_cpu_out.bin ../../data/missile.domn.0.2M 1>> runtime_4.txt 2>> profile_4_oao.txt`  #OAO

   `nvprof --csv --print-gpu-trace ./OAO_manual_pre.bin ../../data/cfd/fvcorr.domn.097K 1>> runtime_6.txt 2>> profile_6_manual.txt`  #manual
   `nvprof --csv --print-gpu-trace ./OAO_manual_pre.bin ../../data/cfd/fvcorr.domn.193K 1>> runtime_5.txt 2>> profile_5_manual.txt`  #manual
   `nvprof --csv --print-gpu-trace ./OAO_manual_pre.bin ../../data/cfd/missile.domn.0.2M 1>> runtime_4.txt 2>> profile_4_manual.txt`  #manual


done
