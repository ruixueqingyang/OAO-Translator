#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding/1119

# make FDTD-2D

cd /home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE0="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_3_function_omp.txt"
FILE1="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_3_function.txt"
FILE4="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_3_dawncc_function.txt"
FILE5="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_3_oao_function.txt"
# FILE6="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_3_manual.txt"

FILE00="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_2_function_omp.txt"
FILE11="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_2_function.txt"
FILE44="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_2_dawncc_function.txt"
FILE55="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_2_oao_function.txt"
# FILE66="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_2_manual.txt"

FILE000="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_1_function_omp.txt"
FILE111="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/runtime_1_function.txt"
FILE444="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_1_dawncc_function.txt"
FILE555="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_1_oao_function.txt"
# FILE666="/home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D/profile_1_manual.txt"

# rm $FILE0 $FILE1  $FILE4 $FILE5 # $FILE6
# rm $FILE00 $FILE11  $FILE44 $FILE55 # $FILE66
# rm $FILE000 $FILE111  $FILE444 $FILE555 # $FILE666

rm $FILE1  $FILE4 $FILE5 # $FILE6
rm $FILE11  $FILE44 $FILE55 # $FILE66
rm $FILE111  $FILE444 $FILE555 # $FILE666

#最小数据量
make clean

make DAWNKERNEL  #DawnCC
make DEBUG=-D_DEBUG_ EXPENSES=-DEXPENSES OAOKERNEL   #OAO
# make DEBUG=-D_DEBUG_ OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWNKERNEL.bin 1>> /dev/null 2>> profile_3_dawncc_function.txt` #DawnCC
  `./DAWNKERNEL.bin 1>> runtime_3_function.txt` #DawnCC
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAOKERNEL.bin 1>> /dev/null 2>> profile_3_oao_function.txt`  
  `./OAOKERNEL.bin 1>> runtime_3_function.txt`
  # `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>> runtime_3.txt 2>> profile_3_manual.txt`  #OAO
done

# make DEBUG=-D_DEBUG_ OMPKERNEL
# for ((i=1; i<=$times; i ++))
# do
#   ./OMPKERNEL.bin 1>> runtime_3_function_omp.txt  #OAO
# done

#中等数据量
make clean

make DAWNKERNEL2  #DawnCC
make DEBUG=-D_DEBUG_2 EXPENSES=-DEXPENSES OAOKERNEL   #OAO
# make DEBUG=-D_DEBUG_2 OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWNKERNEL2.bin 1>> /dev/null 2>> profile_2_dawncc_function.txt` #DawnCC
  `./DAWNKERNEL2.bin 1>> runtime_2_function.txt` #DawnCC
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAOKERNEL.bin 1>> /dev/null 2>> profile_2_oao_function.txt`  #OAO
  `./OAOKERNEL.bin 1>> runtime_2_function.txt`  #OAO
  # `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>> runtime_2.txt 2>> profile_2_manual.txt`  #OAO
done

# make DEBUG=-D_DEBUG_2 OMPKERNEL
# for ((i=1; i<=$times; i ++))
# do
#   ./OMPKERNEL.bin 1>> runtime_2_function_omp.txt  #OAO
# done

#最大数据量
make clean

make DAWNKERNEL1  #DawnCC
make DEBUG=-D_DEBUG_1 EXPENSES=-DEXPENSES OAOKERNEL   #OAO
# make DEBUG=-D_DEBUG_1 OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWNKERNEL1.bin 1>> /dev/null 2>> profile_1_dawncc_function.txt` #DawnCC
  `./DAWNKERNEL1.bin 1>> runtime_1_function.txt` #DawnCC
  `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAOKERNEL.bin 1>> /dev/null 2>> profile_1_oao_function.txt`  #OAO
  `./OAOKERNEL.bin 1>> runtime_1_function.txt`  #OAO
  # `nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>> runtime_1.txt 2>> profile_1_manual.txt`  #OAO
done

# make DEBUG=-D_DEBUG_1 OMPKERNEL
# for ((i=1; i<=$times; i ++))
# do
#   ./OMPKERNEL.bin 1>> runtime_1_function_omp.txt  #OAO
# done

make clean
