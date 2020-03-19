/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: RunTime.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190711
  Description: 实现 一些通用的函数
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1. main: 主函数
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/
#include "RunTime.h"
#include <assert.h>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <system_error>
#include <vector>

#include <sys/time.h>
#include <unistd.h>

// #define _DEBUG_

// #define EXPENSES

// 用来描述数据一致性状态
// bit2: device 是否最新, bit1: host 是否最新, bit0: 是否 map 到 device
enum SYNC_STATE : unsigned int {
    SYNC_UNINIT = 0,
    HOST_ONLY = Bit1_1,           // 说明当前变量只在 host 内存中, 没有被发送到 device 的数据环境
    HOST_NEW = (Bit1_1 | Bit0_1), // host 内存中的数据更新
    DEVICE_NEW = (Bit2_1 | Bit0_1),    // device 的数据环境中的数据更新
    SYNC = (Bit2_1 | Bit1_1 | Bit0_1), // host/device 上的数据同步

    MASK = (Bit2_1 | Bit1_1 | Bit0_1) // 掩码, 用来参与 &运算, 检验同步状态是否协调
};

STATE_CONSTR NoneTrans((Bit2_1 | Bit1_1 | Bit0_1), 0);
STATE_CONSTR EnterAlloc((Bit2_1 | Bit1_1 | Bit0_1), Bit0_1);
STATE_CONSTR EnterTo((Bit2_1 | Bit1_1 | Bit0_1), (Bit2_1 | Bit0_1));
STATE_CONSTR UpdateTo((Bit2_1 | Bit1_1 | Bit0_1), Bit2_1);
STATE_CONSTR UpdateFrom((Bit2_1 | Bit1_1 | Bit0_1), Bit1_1);
STATE_CONSTR ExitDelete(Bit1_1, Bit1_1);

STATE_CONSTR SEQWrite((Bit1_1 | Bit0_1), Bit1_1); // 表示 host 写
STATE_CONSTR OMPWrite((Bit2_1 | Bit0_1), Bit2_1); // 表示 device 写

// wfr 20190712 用来保存 malloc内存空间/静态定义的数组 的范围, 元素个数, 元素size
// 同步状态 等
class MEM_BLOCK {
public:
    unsigned long long int Begin, End; // 内存范围是 [Begin, End] 的闭区间
    unsigned long long int ElementNum;
    unsigned long long int ElementSize;
    unsigned long long int Length;

    // bit2: device 是否最新, bit1: host 是否最新, bit0: 是否 map 到 device
    unsigned int SyncState;

    bool isMalloced; // 标识是 malloc 的内存空间, 还是静态定义数组

    MEM_BLOCK(unsigned long long int in_Begin, unsigned long long int in_length, unsigned long long int in_ElementSize,
              bool in_isMalloced) {
        Begin = in_Begin;
        Length = in_length;
        ElementSize = in_ElementSize;
        ElementNum = in_length / ElementSize;
        End = Begin + in_length - 1;
        isMalloced = in_isMalloced;
        SyncState = SYNC_STATE::HOST_ONLY;
#ifdef _DEBUG_
        std::cout << "MEM_BLOCK: SyncState = 0x" << std::hex << SyncState << std::endl;
#endif
    }

    // 重载 == , 用 find 函数在 vector 里 按照 ptr 搜索时, 会用到这个
    bool operator==(unsigned long long int ptr) {
        // 判断指针是否在当前内存区域中
        if (Begin <= ptr && ptr <= End) {
            return true;
        }
        return false;
    }

    void ExeStTrans(const STATE_CONSTR &StTransFunc) {
        SyncState = (SyncState & StTransFunc.ZERO) | StTransFunc.ONE;
#ifdef _DEBUG_
        std::cout << "ExeStTrans: SyncState = 0x" << std::hex << SyncState << std::endl;
#endif
        return;
    }
};

std::vector<MEM_BLOCK> OAOMemEnv;

double expenseTime = 0.000000; //开销（DataTrans的开销时间）
double OAOtime = 0.000000;     //开销（除DataTrans以外的时间）

struct timeval timeStart, timeEnd, timeStart1, timeEnd1;

double runTime = 0.000000;
//host to device
int H2D_number = 0;           //次数
std::vector<long> H2D_length; //字节数
double H2D_time = 0.000000;   //时间
//device to host
int D2H_number = 0;           //次数
std::vector<long> D2H_length; //字节数
double D2H_time = 0.000000;   //时间

/*
    gettimeofday(&timeStart, NULL);
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    printf("runTime is %lf s\n", runTime);
*/


int InWhinchMem(void *ptr) {
    if (ptr == NULL) {
        return -1;
    }
    for (int i = 0; i < OAOMemEnv.size(); ++i) {
        if (OAOMemEnv[i] == (unsigned long long int)ptr) {
            return i;
        }
    }
    return -1;
}

void *OAONewInfo(void* ptr, unsigned long long int ElementSize, unsigned long long int ElementNum) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    OAOMemEnv.emplace_back((unsigned long long int)ptr, ElementSize*ElementNum, ElementSize, 1);
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return ptr;
}

// 删除保存的 malloc 信息
void OAODeleteInfo(void *ptr) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    if (ptr == NULL) {
        return;
    }
    int index = InWhinchMem(ptr);
    if (index >= 0) {
        OAOMemEnv.erase(OAOMemEnv.begin() + index);
    }
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return;
}

// 保存静态定义的数组的信息
void OAOArrayInfo(void *ptr, unsigned long long int length, unsigned long long int ElementSize) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    if (ptr == NULL) {
        return;
    }
    int index = InWhinchMem(ptr);
    if (index < 0) {
        OAOMemEnv.emplace_back((unsigned long long int)ptr, length, ElementSize, 0);
    }
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return;
}

// 保存 malloc 信息
void OAOSaveMallocInfo(void *ptr, unsigned long long int length, unsigned long long int ElementSize) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    if (ptr == NULL) {
        return;
    }
    OAOMemEnv.emplace_back((unsigned long long int)ptr, length, ElementSize, 1);
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return;
}

// 分配内存, 保存 malloc 信息
void *OAOMalloc(unsigned long long int length) {
    void *ptr = malloc(length);
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    
    if (ptr == NULL) {
        std::cout << "OAOMalloc 错误: malloc 失败!" << std::endl;
        exit(1);
    }
    OAOMemEnv.emplace_back((unsigned long long int)ptr, length, 1, 1);
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return ptr;
}

// 释放内存, 删除保存的 malloc 信息
void OAOFree(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    free(ptr);
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    
    int index = InWhinchMem(ptr);
    if (index >= 0) {
        OAOMemEnv.erase(OAOMemEnv.begin() + index);
    }
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return;
}

// 删除保存的 malloc 信息
void OAODeleteMallocInfo(void *ptr) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    if (ptr == NULL) {
        return;
    }

    int index = InWhinchMem(ptr);
    if (index >= 0) {
        OAOMemEnv.erase(OAOMemEnv.begin() + index);
    }
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    OAOtime += runTime;
#endif
    return;
}

// 进行数据传输, 并更新同步状态
void OAODataTrans(void *ptr, STATE_CONSTR StReq) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    if (ptr == NULL) {
        return;
    }
    int index = InWhinchMem(ptr);
    if (index < 0) {
#ifdef _DEBUG_
        std::cout << "OAODataTrans 警告: 运行时没找到指针所在的内存区域, 无法进行数据传输!" << std::endl;
#endif
        return;
    }

    // 计算从 SyncState 到 StReq 所需的状态转移函数
    unsigned int ZERO = ~((OAOMemEnv[index].SyncState ^ StReq.ZERO) & (~StReq.ZERO));
    unsigned int ONE = (OAOMemEnv[index].SyncState ^ StReq.ONE) & StReq.ONE;
    STATE_CONSTR StTransFunc(ZERO, ONE);

    char *BeginPtr = (char *)OAOMemEnv[index].Begin;
    unsigned long long int Length = OAOMemEnv[index].Length;
    

    if (StTransFunc == NoneTrans) {
#ifdef _DEBUG_
        std::cout << "OAODataTrans: 已达到同步状态需求, 不进行传输" << std::endl;
#endif
        return;
    } else if (StTransFunc == EnterAlloc) {
#pragma omp target enter data map(alloc : BeginPtr[:Length])
        OAOMemEnv[index].ExeStTrans(StTransFunc);
    } else if (StTransFunc == EnterTo) {
#ifdef _DEBUG_
        std::cout << "OAODataTrans: enter data map( to: BeginPtr[:Length] ) " << std::endl;
        std::cout << "BeginPtr = 0x" << std::hex << (unsigned long long)BeginPtr << std::endl;
        std::cout << "Length = " << std::dec << Length << std::endl;
#endif
#ifdef EXPENSES
    gettimeofday(&timeStart1, NULL);
#endif
#pragma omp target enter data map(to : BeginPtr[:Length])
        OAOMemEnv[index].ExeStTrans(StTransFunc);
#ifdef EXPENSES
    H2D_number ++;
    H2D_length.push_back(Length);
    gettimeofday(&timeEnd1, NULL);
    runTime = (timeEnd1.tv_sec - timeStart1.tv_sec) + (double)(timeEnd1.tv_usec - timeStart1.tv_usec) / 1000000;
    H2D_time += runTime;
#endif
    } else if (StTransFunc == UpdateTo) {
#ifdef EXPENSES
    gettimeofday(&timeStart1, NULL);
#endif
#pragma omp target update to(BeginPtr[:Length])
        OAOMemEnv[index].ExeStTrans(StTransFunc);
#ifdef EXPENSES
    H2D_number ++;
    H2D_length.push_back(Length);
    gettimeofday(&timeEnd1, NULL);
    runTime = (timeEnd1.tv_sec - timeStart1.tv_sec) + (double)(timeEnd1.tv_usec - timeStart1.tv_usec) / 1000000;
    H2D_time += runTime;
#endif
    } else if (StTransFunc == UpdateFrom) {
#ifdef _DEBUG_
        std::cout << "OAODataTrans: update from( BeginPtr[:Length] ) " << std::endl;
        std::cout << "BeginPtr = 0x" << std::hex << (unsigned long long)BeginPtr << std::endl;
        std::cout << "Length = " << std::dec << Length << std::endl;
#endif
#ifdef EXPENSES
    gettimeofday(&timeStart1, NULL);
#endif
#pragma omp target update from(BeginPtr[:Length])
        OAOMemEnv[index].ExeStTrans(StTransFunc);
#ifdef EXPENSES
    D2H_number ++;
    D2H_length.push_back(Length);
    gettimeofday(&timeEnd1, NULL);
    runTime = (timeEnd1.tv_sec - timeStart1.tv_sec) + (double)(timeEnd1.tv_usec - timeStart1.tv_usec) / 1000000;
    D2H_time += runTime;
#endif
    } else if (StReq == ExitDelete) {
#pragma omp target exit data map(delete : BeginPtr[:Length])
        // iter->ExeStTrans(StTransFunc);
        OAODeleteMallocInfo(ptr);
    } else {
#ifdef _DEBUG_
        std::cout << "OAODataTrans 警告: 非法的状态转换函数, 无法进行数据传输!" << std::endl;
        std::cout << "OAODataTrans 警告: ZERO = 0x" << std::hex << StTransFunc.ZERO << ", ONE = 0x" << std::hex
                  << StTransFunc.ONE << std::endl;
#endif
        return;
    }
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    expenseTime += runTime;
#endif
    return;
}

// 进行同步状态转换
void OAOStTrans(void *ptr, STATE_CONSTR StTrans) {
#ifdef EXPENSES
    gettimeofday(&timeStart, NULL);
#endif
    if (ptr == NULL) {
        return;
    }
    STATE_CONSTR StReqUninit(ST_REQ_UNINIT);
    STATE_CONSTR StReqDiverg(ST_REQ_DIVERG);
    if (StTrans == StReqUninit || StTrans == StReqDiverg) {
#ifdef _DEBUG_
        std::cout << "OAOStTrans 警告: 非法的状态转换函数, 无法进行同步状态转换!" << std::endl;
#endif
    }
    int index = InWhinchMem(ptr);
    if (index < 0) {
#ifdef _DEBUG_
        std::cout << "OAOStTrans 警告: 运行时没找到指针所在的内存区域, 无法进行同步状态转换!" << std::endl;
#endif
        return;
    }
    OAOMemEnv[index].ExeStTrans(StTrans);
#ifdef EXPENSES
    gettimeofday(&timeEnd, NULL);
    runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
    expenseTime += runTime;
#endif
}

// 更新同步状态
void OAOSEQWrite(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    int index = InWhinchMem(ptr);
    if (index < 0) {
#ifdef _DEBUG_
        std::cout << "OAOSEQWrite 警告: 运行时没找到指针所在的内存区域, 无法进行同步状态转换!" << std::endl;
#endif
        return;
    }

    OAOMemEnv[index].ExeStTrans(SEQWrite);

#ifdef _DEBUG_
    std::cout << "OAOSEQWrite: 写后同步状态 SyncState = 0x" << std::hex << OAOMemEnv[index].SyncState << std::endl;
#endif
    return;
}

// 更新同步状态
void OAOOMPWrite(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    int index = InWhinchMem(ptr);
    if (index < 0) {
#ifdef _DEBUG_
        std::cout << "OAOOMPWrite 警告: 运行时没找到指针所在的内存区域, 无法进行同步状态转换!" << std::endl;
#endif
        return;
    }

    OAOMemEnv[index].ExeStTrans(OMPWrite);

#ifdef _DEBUG_
    std::cout << "OAOOMPWrite: 写后同步状态 SyncState = 0x" << std::hex << OAOMemEnv[index].SyncState << std::endl;
#endif

    return;
}

// 时间开销
void OAOExpenseTime(){
    std::cout << "OAO Expense Time（except DataTrans） is:" << std::dec << OAOtime << "s" << std::endl;
    std::cout << "OAO Expense Time is:" << std::dec << expenseTime + OAOtime << "s" << std::endl;
    return;
}

// 数据传输信息
void DataTransInfo(){
    std::cout << "Host to Device number of times is:" << std::dec << H2D_number << std::endl;
    long all_H2DLength = 0;
    for(auto i = 0; i < H2D_length.size(); ++i){
        all_H2DLength += H2D_length[i];
    }
    std::cout << "Host to Device number of byte is:" << std::dec << all_H2DLength << std::endl;
    std::cout << "Host to Device time is:" << std::dec << H2D_time + OAOtime << "s" << std::endl;
    

    std::cout << "Device to Host number of times is:" << std::dec << D2H_number << std::endl;
    long all_D2HLength = 0;
    for(auto i = 0; i < D2H_length.size(); ++i){
        all_D2HLength += D2H_length[i];
    }
    std::cout << "Device to Host number of byte is:" << std::dec << all_D2HLength << std::endl;
    std::cout << "Device to Host time is:" << std::dec << D2H_time + OAOtime << "s" << std::endl;

    return;
}