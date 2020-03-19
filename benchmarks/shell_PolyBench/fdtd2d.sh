#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding/1119

# make FDTD-2D

cd /home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE0="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_3_omp.txt"
FILE1="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_3.txt"
FILE4="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_3_dawncc.txt"
FILE5="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_3_oao.txt"
FILE6="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_3_manual.txt"

FILE00="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_2_omp.txt"
FILE11="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_2.txt"
FILE44="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_2_dawncc.txt"
FILE55="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_2_oao.txt"
FILE66="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_2_manual.txt"

FILE000="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_1_omp.txt"
FILE111="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_1.txt"
FILE444="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_1_dawncc.txt"
FILE555="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_1_oao.txt"
FILE666="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_1_manual.txt"

rm $FILE0 $FILE1  $FILE4 $FILE5 $FILE6 
rm $FILE00 $FILE11  $FILE44 $FILE55 $FILE66
rm $FILE000 $FILE111  $FILE444 $FILE555 $FILE666

#最小数据量
make clean

make DAWN  #DawnCC
make DEBUG=-D_DEBUG_ EXPENSES=-DEXPENSES OAO   #OAO
make DEBUG=-D_DEBUG_ OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN.bin 1>> /dev/null 2>> profile_3_dawncc.txt` #DawnCC
  `./DAWN.bin 1>> runtime_3.txt` #DawnCC
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO.bin 1>> /dev/null 2>> profile_3_oao.txt`  #OAO
  `./OAO.bin 1>> runtime_3.txt`  #OAO
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>> /dev/null 2>> profile_3_manual.txt`  #OAO
  `./OAO_manual.bin 1>> runtime_3.txt`  #OAO
done

make DEBUG=-D_DEBUG_ OMP
for ((i=1; i<=$times; i ++))
do
  ./OMP.bin 1>> runtime_3_omp.txt  #OAO
done

#中等数据量
make clean

make DAWN_2  #DawnCC
make DEBUG=-D_DEBUG_2 EXPENSES=-DEXPENSES OAO   #OAO
make DEBUG=-D_DEBUG_2 OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_2.bin 1>> /dev/null 2>> profile_2_dawncc.txt` #DawnCC
  `./DAWN_2.bin 1>> runtime_2.txt` #DawnCC
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO.bin 1>> /dev/null 2>> profile_2_oao.txt` #OAO
  `./OAO.bin 1>> runtime_2.txt` #OAO
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>> /dev/null 2>> profile_2_manual.txt` #OAO
  `./OAO_manual.bin 1>> runtime_2.txt` #OAO
done

make DEBUG=-D_DEBUG_2 OMP
for ((i=1; i<=$times; i ++))
do
  ./OMP.bin 1>> runtime_2_omp.txt  #OAO
done

#最大数据量
make clean

make DAWN_1  #DawnCC
make DEBUG=-D_DEBUG_1 EXPENSES=-DEXPENSES OAO   #OAO
make DEBUG=-D_DEBUG_1 OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_1.bin 1>> /dev/null 2>> profile_1_dawncc.txt` #DawnCC
  `./DAWN_1.bin 1>> runtime_1.txt` #DawnCC
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO.bin 1>> /dev/null 2>> profile_1_oao.txt`  #OAO
  `./OAO.bin 1>> runtime_1.txt`  #OAO
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>> /dev/null 2>> profile_1_manual.txt`  #OAO
  `./OAO_manual.bin 1>> runtime_1.txt`  #OAO
done

make DEBUG=-D_DEBUG_1 OMP
for ((i=1; i<=$times; i ++))
do
  ./OMP.bin 1>> runtime_1_omp.txt  #OAO
done

make clean
