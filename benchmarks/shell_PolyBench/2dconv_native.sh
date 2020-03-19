#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding/1119

# make 2DCONV

cd /home/wfr/work/Coding/OAO_2080Ti/plain/2DCONV

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

RUNTIME_DAWNCC_NATIVE_3="runtime_3_dawncc_native.txt"
PROFILE_DAWNCC_NATIVE_3="profile_3_dawncc_native.txt"

RUNTIME_DAWNCC_NATIVE_2="runtime_2_dawncc_native.txt"
PROFILE_DAWNCC_NATIVE_2="profile_2_dawncc_native.txt"

RUNTIME_DAWNCC_NATIVE_1="runtime_1_dawncc_native.txt"
PROFILE_DAWNCC_NATIVE_1="profile_1_dawncc_native.txt"

#三种数据量
FLAG_3=1
FLAG_2=1
FLAG_1=1

#运行时间 / 传输分析
FLAG_RUNTIME=1
FLAG_PROFILE=1

make clean

#最小数据量
if (($FLAG_3 != 0)); then
  echo "Date Size 3"
  make DAWN_native

  if (($FLAG_RUNTIME != 0)); then
    echo "$RUNTIME_DAWNCC_NATIVE_3"
    rm $RUNTIME_DAWNCC_NATIVE_3
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      ./DAWN_native.bin 1>>$RUNTIME_DAWNCC_NATIVE_3
    done
  fi

  if (($FLAG_PROFILE != 0)); then
    echo "$PROFILE_DAWNCC_NATIVE_3"
    rm $PROFILE_DAWNCC_NATIVE_3
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_native.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_NATIVE_3
    done
  fi

fi

#中等数据量
if (($FLAG_2 != 0)); then
  echo "Date Size 2"
  make DAWN_2_native

  if (($FLAG_RUNTIME != 0)); then
    echo "$RUNTIME_DAWNCC_NATIVE_2"
    rm $RUNTIME_DAWNCC_NATIVE_2
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      ./DAWN_2_native.bin 1>>$RUNTIME_DAWNCC_NATIVE_2
    done
  fi

  if (($FLAG_PROFILE != 0)); then
    echo "$PROFILE_DAWNCC_NATIVE_2"
    rm $PROFILE_DAWNCC_NATIVE_2
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_2_native.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_NATIVE_2
    done
  fi

fi

#最大数据量
if (($FLAG_1 != 0)); then
  echo "Date Size 1"
  make DAWN_1_native

  if (($FLAG_RUNTIME != 0)); then
    echo "$RUNTIME_DAWNCC_NATIVE_1"
    rm $RUNTIME_DAWNCC_NATIVE_1
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      ./DAWN_1_native.bin 1>>$RUNTIME_DAWNCC_NATIVE_1
    done
  fi

  if (($FLAG_PROFILE != 0)); then
    echo "$PROFILE_DAWNCC_NATIVE_1"
    rm $PROFILE_DAWNCC_NATIVE_1
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_1_native.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_NATIVE_1
    done
  fi

fi

make clean
