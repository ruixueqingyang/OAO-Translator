#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make cfd

cd /home/anjushi/work/par/Rodinia_3.1/openmp/cfd

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/runtime_3.txt"
FILE4="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_3_omp.txt"
FILE5="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_3_oao.txt"
FILE6="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_3_manual.txt"

FILE11="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/runtime_2.txt"
FILE44="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_2_omp.txt"
FILE55="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_2_oao.txt"
FILE66="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_2_manual.txt"

FILE111="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/runtime_1.txt"
FILE444="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_1_omp.txt"
FILE555="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_1_oao.txt"
FILE666="/home/anjushi/work/par/Rodinia_3.1/openmp/cfd/profile_1_manual.txt"

rm -rf $FILE1  $FILE4  $FILE5  $FILE6
rm -rf $FILE11  $FILE44  $FILE55  $FILE66
rm -rf $FILE111  $FILE444  $FILE555  $FILE666

make clean

make euler3d_cpu  #OMP
make EXPENSES=-DEXPENSES euler3d_cpu_out   #OAO
make OAO_manual   #manual

for ((i=1; i<=$times; i ++))
do  
   `nvprof --csv --print-gpu-trace ./euler3d_cpu ../../data/cfd/fvcorr.domn.097K 1>> /dev/null 2>> profile_3_omp.txt`  #OAO
   `./euler3d_cpu ../../data/cfd/fvcorr.domn.097K 1>> runtime_3.txt`
   `nvprof --csv --print-gpu-trace ./euler3d_cpu ../../data/cfd/fvcorr.domn.193K 1>> /dev/null 2>> profile_2_omp.txt`  #OAO
   `./euler3d_cpu ../../data/cfd/fvcorr.domn.193K 1>> runtime_2.txt`
   `nvprof --csv --print-gpu-trace ./euler3d_cpu ../../data/cfd/missile.domn.0.2M 1>> /dev/null 2>> profile_1_omp.txt`  #OAO
   `./euler3d_cpu ../../data/cfd/missile.domn.0.2M 1>> runtime_1.txt`

   `nvprof --csv --print-gpu-trace ./euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.097K 1>> /dev/null 2>> profile_3_oao.txt`  #OAO
   `./euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.097K 1>> runtime_3.txt`
   `nvprof --csv --print-gpu-trace ./euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.193K 1>> /dev/null 2>> profile_2_oao.txt`  #OAO
   `./euler3d_cpu_out.bin ../../data/cfd/fvcorr.domn.193K 1>> runtime_2.txt`
   `nvprof --csv --print-gpu-trace ./euler3d_cpu_out.bin ../../data/cfd/missile.domn.0.2M 1>> /dev/null 2>> profile_1_oao.txt`  #OAO
   `./euler3d_cpu_out.bin ../../data/cfd/missile.domn.0.2M 1>> runtime_1.txt`

   `nvprof --csv --print-gpu-trace ./OAO_manual.bin ../../data/cfd/fvcorr.domn.097K 1>> /dev/null 2>> profile_3_manual.txt`  #manual
   `./OAO_manual.bin ../../data/cfd/fvcorr.domn.097K 1>> runtime_3.txt`
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin ../../data/cfd/fvcorr.domn.193K 1>> /dev/null 2>> profile_2_manual.txt`  #manual
   `./OAO_manual.bin ../../data/cfd/fvcorr.domn.193K 1>> runtime_2.txt`
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin ../../data/cfd/missile.domn.0.2M 1>> /dev/null 2>> profile_1_manual.txt`  #manual
   `./OAO_manual.bin ../../data/cfd/missile.domn.0.2M 1>> runtime_1.txt`

done
