#!/bin/bash

times=$1

# cd /home/anjushi/work/coding/1119

# make FDTD-2D

cd /home/wfr/work/Coding/OAO_2080Ti/plain/FDTD-2D

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

RUNTIME_OMP_3="runtime_3_omp_function.txt"
RUNTIME_OAO_3="runtime_3_oao_function.txt"
RUNTIME_DAWNCC_3="runtime_3_dawncc_function.txt"
PROFILE_OAO_3="profile_3_oao_function.txt"
PROFILE_DAWNCC_3="profile_3_dawncc_function.txt"

RUNTIME_OMP_2="runtime_2_omp_function.txt"
RUNTIME_OAO_2="runtime_2_oao_function.txt"
RUNTIME_DAWNCC_2="runtime_2_dawncc_function.txt"
PROFILE_OAO_2="profile_2_oao_function.txt"
PROFILE_DAWNCC_2="profile_2_dawncc_function.txt"

RUNTIME_OMP_1="runtime_1_omp_function.txt"
RUNTIME_OAO_1="runtime_1_oao_function.txt"
RUNTIME_DAWNCC_1="runtime_1_dawncc_function.txt"
PROFILE_OAO_1="profile_1_oao_function.txt"
PROFILE_DAWNCC_1="profile_1_dawncc_function.txt"

#三种数据量
FLAG_3=1
FLAG_2=1
FLAG_1=1

#运行时间 / 传输分析
FLAG_RUNTIME=1
FLAG_PROFILE=1

#不同程序版本
FLAG_OMP=0
FLAG_OAO=1
FLAG_DAWNCC=1

make clean

#最小数据量
if (($FLAG_3 != 0)); then

  #OAO
  if (($FLAG_OAO != 0)); then
    make DEBUG=-D_DEBUG_ EXPENSES=-DEXPENSES OAOKERNEL

    if (($FLAG_RUNTIME != 0)); then
      rm $RUNTIME_OAO_3
      for ((i = 1; i <= $times; i++)); do
        ./OAOKERNEL.bin 1>>$RUNTIME_OAO_3
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      rm $PROFILE_OAO_3
      for ((i = 1; i <= $times; i++)); do
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAOKERNEL.bin 1>>/dev/null 2>>$PROFILE_OAO_3
      done
    fi

  fi

  #DawnCC
  if (($FLAG_DAWNCC != 0)); then
    make DAWNKERNEL

    if (($FLAG_RUNTIME != 0)); then
      rm $RUNTIME_DAWNCC_3
      for ((i = 1; i <= $times; i++)); do
        ./DAWNKERNEL.bin 1>>$RUNTIME_DAWNCC_3
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      rm $PROFILE_DAWNCC_3
      for ((i = 1; i <= $times; i++)); do
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWNKERNEL.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_3
      done
    fi
  fi

  #OMP
  if (($FLAG_OMP != 0)) && (($FLAG_RUNTIME != 0)); then
    make DEBUG=-D_DEBUG_ OMPKERNEL
    rm $RUNTIME_OMP_3
    for ((i = 1; i <= $times; i++)); do
      ./OMPKERNEL.bin 1>>$RUNTIME_OMP_3
    done
  fi

fi

#中等数据量
if (($FLAG_2 != 0)); then

  #OAO
  if (($FLAG_OAO != 0)); then
    make DEBUG=-D_DEBUG_2 EXPENSES=-DEXPENSES OAOKERNEL

    if (($FLAG_RUNTIME != 0)); then
      rm $RUNTIME_OAO_2
      for ((i = 1; i <= $times; i++)); do
        ./OAOKERNEL.bin 1>>$RUNTIME_OAO_2
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      rm $PROFILE_OAO_2
      for ((i = 1; i <= $times; i++)); do
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAOKERNEL.bin 1>>/dev/null 2>>$PROFILE_OAO_2
      done
    fi
  fi

  #DawnCC
  if (($FLAG_DAWNCC != 0)); then
    make DAWNKERNEL2

    if (($FLAG_RUNTIME != 0)); then
      rm $RUNTIME_DAWNCC_2
      for ((i = 1; i <= $times; i++)); do
        ./DAWNKERNEL2.bin 1>>$RUNTIME_DAWNCC_2
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      rm $PROFILE_DAWNCC_2
      for ((i = 1; i <= $times; i++)); do
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWNKERNEL2.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_2
      done
    fi
  fi

  #OMP
  if (($FLAG_OMP != 0)) && (($FLAG_RUNTIME != 0)); then
    make DEBUG=-D_DEBUG_2 OMPKERNEL
    rm $RUNTIME_OMP_2
    for ((i = 1; i <= $times; i++)); do
      ./OMPKERNEL.bin 1>>$RUNTIME_OMP_2
    done
  fi

fi

#最大数据量
if (($FLAG_1 != 0)); then

  #OAO
  if (($FLAG_OAO != 0)); then
    make DEBUG=-D_DEBUG_1 EXPENSES=-DEXPENSES OAOKERNEL

    if (($FLAG_RUNTIME != 0)); then
      rm $RUNTIME_OAO_1
      for ((i = 1; i <= $times; i++)); do
        ./OAOKERNEL.bin 1>>$RUNTIME_OAO_1
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      rm $PROFILE_OAO_1
      for ((i = 1; i <= $times; i++)); do
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAOKERNEL.bin 1>>/dev/null 2>>$PROFILE_OAO_1
      done
    fi
  fi

  #DawnCC
  if (($FLAG_DAWNCC != 0)); then
    make DAWNKERNEL1

    if (($FLAG_RUNTIME != 0)); then
      rm $RUNTIME_DAWNCC_1
      for ((i = 1; i <= $times; i++)); do
        ./DAWNKERNEL1.bin 1>>$RUNTIME_DAWNCC_1
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      rm $PROFILE_DAWNCC_1
      for ((i = 1; i <= $times; i++)); do
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWNKERNEL1.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_1
      done
    fi
  fi

  #OMP
  if (($FLAG_OMP != 0)) && (($FLAG_RUNTIME != 0)); then
    make DEBUG=-D_DEBUG_1 OMPKERNEL
    rm $RUNTIME_OMP_1
    for ((i = 1; i <= $times; i++)); do
      ./OMPKERNEL.bin 1>>$RUNTIME_OMP_1
    done
  fi

fi

make clean
