/***************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: RunTime.h
       Author: 瑞雪轻飏
      Version: 0.1
Creation Date: 20190711
  Description: 
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
***************************************************************************/

#ifndef RUN_TIME_H
#define RUN_TIME_H

#include <stdio.h>
#include <stdlib.h>
#include <system_error>
#include <string.h>
#include <vector>
#include <iostream>
#include <assert.h>
#include <malloc.h>
#include <cuda_runtime.h>

#define Bit0_0  (0)
#define Bit0_1  (1)
#define Bit1_0  (0)
#define Bit1_1  (1<<1)
#define Bit2_0  (0)
#define Bit2_1  (1<<2)
#define BitMask (Bit2_1 | Bit1_1 | Bit0_1)

// 入口状态需求未确定
#define ST_REQ_UNINIT 0, 0
// 不同执行路径的入口状态需求出现分歧 divergence
#define ST_REQ_DIVERG 0, (Bit2_1 | Bit1_1 | Bit0_1)
// 对入口状态没有要求
#define ST_REQ_NONE (Bit2_1 | Bit1_1 | Bit0_1), 0
#define ST_REQ_HOST_ONLY Bit1_1, Bit1_1
// wfr 20191207 为了不再使用 alloc 操作, 减小开销
// #define ST_REQ_HOST_NEW (Bit2_1 | Bit1_1 | Bit0_1), (Bit1_1 | Bit0_1)
#define ST_REQ_HOST_NEW (Bit2_1 | Bit1_1 | Bit0_1), (Bit1_1)
#define ST_REQ_DEVICE_NEW (Bit2_1 | Bit1_1 | Bit0_1), (Bit2_1 | Bit0_1)
#define ST_REQ_IN_DEVICE (Bit2_1 | Bit1_1 | Bit0_1), Bit0_1
#define ST_REQ_SYNC (Bit2_1 | Bit1_1 | Bit0_1), (Bit2_1 | Bit1_1 | Bit0_1)

// 状态转换函数未确定
#define ST_TRANS_UNINIT 0, 0
// 不同执行路径的状态转换函数出现分歧 divergence
#define ST_TRANS_DIVERG 0, (Bit2_1 | Bit1_1 | Bit0_1)
// 不进行同步状态转换
#define ST_TRANS_NONE (Bit2_1 | Bit1_1 | Bit0_1), 0
#define ST_TRANS_HOST_READ (Bit2_1 | Bit1_1 | Bit0_1), 0
#define ST_TRANS_DEVICE_READ (Bit2_1 | Bit1_1 | Bit0_1), 0
#define ST_TRANS_HOST_WRITE (Bit1_1 | Bit0_1), Bit1_1
#define ST_TRANS_DEVICE_WRITE (Bit2_1 | Bit0_1), Bit2_1
#define ST_TRANS_HOST_FREE (Bit2_1 | Bit1_1 | Bit0_1), 0
#define ST_TRANS_SYNC (Bit2_1 | Bit1_1 | Bit0_1), (Bit2_1 | Bit1_1 | Bit0_1)

// wfr 20190718 通过这个宏 定义所有的 StReq 和 StTrans 类型
#define DEFINE_ST_REQ_ST_TRANS \
STATE_CONSTR StReqUninit(ST_REQ_UNINIT); \
STATE_CONSTR StReqDiverg(ST_REQ_DIVERG); \
STATE_CONSTR StReqNone(ST_REQ_NONE); \
STATE_CONSTR StReqHostOnly(ST_REQ_HOST_ONLY); \
STATE_CONSTR StReqHostNew(ST_REQ_HOST_NEW); \
STATE_CONSTR StReqDeviceNew(ST_REQ_DEVICE_NEW); \
STATE_CONSTR StReqInDevice(ST_REQ_IN_DEVICE); \
STATE_CONSTR StReqSync(ST_REQ_SYNC); \
 \
STATE_CONSTR StTransUninit(ST_TRANS_UNINIT); \
STATE_CONSTR StTransDiverg(ST_TRANS_DIVERG); \
STATE_CONSTR StTransNone(ST_TRANS_NONE); \
STATE_CONSTR StTransHostRead(ST_TRANS_HOST_READ); \
STATE_CONSTR StTransDeviceRead(ST_TRANS_DEVICE_READ); \
STATE_CONSTR StTransHostWrite(ST_TRANS_HOST_WRITE); \
STATE_CONSTR StTransDeviceWrite(ST_TRANS_DEVICE_WRITE); \
STATE_CONSTR StTransHostFree(ST_TRANS_HOST_FREE); \
STATE_CONSTR StTransSync(ST_TRANS_SYNC);

// 用来表示对同步状态的约束, 例如: 入口同步状态需求, 状态转换函数
class STATE_CONSTR{
public:
    unsigned int ZERO, ONE;
    
    bool operator==(const STATE_CONSTR& in){
        if((ZERO&BitMask)==(in.ZERO&BitMask) 
            && (ONE&BitMask)==(in.ONE&BitMask)){
            return true;
        }
        return false;
    }
    
    STATE_CONSTR(){
        ZERO = (Bit2_1 | Bit1_1 | Bit0_1);
        ONE = 0;
    }
    
    STATE_CONSTR& init(unsigned int in_ZERO, unsigned int in_ONE){
        ZERO = in_ZERO & BitMask;
        ONE = in_ONE & BitMask;
        return *this;
    }
    
    STATE_CONSTR(unsigned int in_ZERO, unsigned int in_ONE){
        init(in_ZERO, in_ONE);
    }
    
    const STATE_CONSTR& operator=(const STATE_CONSTR& in){
        init(in.ZERO, in.ONE);
        return in;
    }
    
    STATE_CONSTR(const STATE_CONSTR& in){
        init(in.ZERO, in.ONE);
    }
};

// 衡量使用 malloc/cudaMallocHost 的阈值
const unsigned long long int threshold = (unsigned long long int)128 * 1024;

// 标识 内存区域 的分配类型: 0 是静态定义数组; 1 是 malloc() 分配; 2 是cudaMallocHost() 分配; 3 是 new 分配
#define MEM_STATIC_ARRAY 0
#define MEM_MALLOC 1
#define MEM_CUDA 2
#define MEM_NEW 2

// wfr 20200112 重载 new
void* operator new(size_t size);
// wfr 20200112 重载 new[]
void* operator new[](size_t size);

// wfr 20200112 重载 delete
void operator delete(void* mem) noexcept;
// wfr 20200112 重载 delete[]
void operator delete[](void* mem) noexcept;

void* OAONewInfo(void* ptr, unsigned long long int ElementSize, unsigned long long int ElementNum);

void OAODeleteInfo(void *ptr);

// 分配内存, 保存 malloc 信息
void* OAOMalloc(unsigned long long int length);

// 释放内存, 删除保存的 malloc 信息
void OAOFree(void* ptr);

// 保存静态定义的数组的信息
void OAOArrayInfo(void* ptr, unsigned long long int length, unsigned long long int ElementSize);

// // 保存 malloc 信息
// void OAOSaveMallocInfo(void* ptr, unsigned long long int length, unsigned long long int ElementSize);

// // 删除保存的 malloc 信息
// void OAODeleteMallocInfo(void* ptr);

// 进行数据传输, 并更新同步状态
void OAODataTrans(void* ptr, STATE_CONSTR StReq);

// 进行同步状态转换
void OAOStTrans(void* ptr, STATE_CONSTR StTrans);

// 时间开销
void OAOExpenseTime();

// // 数据传输信息
// void DataTransInfo();

#endif // RUN_TIME_H

