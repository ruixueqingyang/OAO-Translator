/***************************************************************************
Copyright(C), 2010-2019, 瑞雪轻飏
     FileName: SeqParNodeBase.h
       Author: 瑞雪轻飏
      Version: 0.1
Creation Date: 20181216
  Description: 串并行图的基类, 本文件中的基类主要实现图相关的功能
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

#ifndef SEQ_PAR_NODE_BASE_H
#define SEQ_PAR_NODE_BASE_H

#include "Clang_LLVM_Inc.h"
#include "SrcRange.h"

using namespace clang;

#define OAO_MALLOC_NAME "OAOMalloc"
#define OAO_FREE_NAME "OAOFree"
#define OAO_NEW_NAME "OAONewInfo"
#define OAO_DELETE_NAME "OAODeleteInfo"
#define OAO_ARRAY_NAME "OAOArrayInfo"
#define OAO_DATA_TRANS "OAODataTrans"
#define OAO_ST_TRANS "OAOStTrans"


#define Bit0_0  (0)
#define Bit0_1  (1)
#define Bit1_0  (0)
#define Bit1_1  (1<<1)
#define Bit2_0  (0)
#define Bit2_1  (1<<2)
#define BitMask  (Bit2_1 | Bit1_1 | Bit0_1)

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

// 用来表示对同步状态的约束, 例如: 入口同步状态需求, 状态转换函数
class STATE_CONSTR{
public:
    unsigned int ZERO, ONE;
    
    bool operator!=(const STATE_CONSTR& in){
        if((ZERO&BitMask)==(in.ZERO&BitMask) 
            && (ONE&BitMask)==(in.ONE&BitMask)){
            return false;
        }
        return true;
    }

    bool operator==(const STATE_CONSTR& in){
        if((ZERO&BitMask)==(in.ZERO&BitMask) 
            && (ONE&BitMask)==(in.ONE&BitMask)){
            return true;
        }
        return false;
    }
    
    STATE_CONSTR(){
        ZERO = 0;
        ONE = 0;
    }
    
    void init(unsigned int in_ZERO, unsigned int in_ONE){
        ZERO = in_ZERO & BitMask;
        ONE = in_ONE & BitMask;
        return;
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


enum SCOPE_BEGIN_END : unsigned int{ BEGIN, END };

// 用来描述数据一致性状态
// bit2: device 是否最新, bit1: host 是否最新, bit0: 是否 map 到 device
enum SYNC_BITS : unsigned int{
    BITS_UNINIT=0,
    BITS_HOST_ONLY=Bit1_1, // 说明当前变量只在 host 内存中, 没有被发送到 device 的数据环境
    BITS_HOST_NEW=(Bit1_1 | Bit0_1), // host 内存中的数据更新
    BITS_DEVICE_NEW=(Bit2_1 | Bit0_1), // device 的数据环境中的数据更新
    BITS_SYNC=(Bit2_1 | Bit1_1 | Bit0_1), // host/device 上的数据同步

    BITS_MASK=(Bit2_1 | Bit1_1 | Bit0_1) // 掩码, 用来参与 &运算, 检验同步状态是否协调
};

// 用来描述数据一致性状态
// 同步状态： 未初始化 同步 host更新 device更新 只存在host上
// 同步状态要求, 结束时同步状态: 同步, host最新, device最新, host-only, 无要求, 不改变
enum SYNC_STATE : unsigned int{ // 设置成不同的位, 方便之后用 & | 来判断状态
    SYNC_UNINIT=0, // 在当前节点的父节点们中, 当前变量的 同步状态 不协调, 或者 同步状态 还没初始化
    HOST_NEW=1, // host 内存中的数据更新
    DEVICE_NEW=1<<1, // device 的数据环境中的数据更新
    SYNC=(1+(1<<1)), // host/device 上的数据同步
    HOST_ONLY=1<<2, // 说明当前变量只在 host 内存中, 没有被发送到 device 的数据环境
    UNUSED=((1<<2)+(1<<3)), // 说明之前的 SEQ/OMP 节点没有使用当前变量
    NO_REQUIREMENT=1<<4, // 对于当前变量的同步状态没有要求
    UNCHANGED=1<<5, // 当前节点没有改变当前变量的同步状态
    IN_DEVICE=1<<6, // 只要求 device 数据环境中有当前变量就行
    MASK=(unsigned int)(0xFFFFFFFF) // 掩码, 用来参与 &运算, 检验同步状态是否协调
};

// 变量引用类型
enum REF_TYPE : unsigned int{ REF_UNINIT=0, READ=1, WRITE=1<<1, READ_WRITE=1<<2, WRITE_READ=1<<3, FREE=1<<4 };

enum NODE_TYPE : unsigned int{ NODE_UNINIT=0, SEQUENTIAL=1, PARALLEL=2};
enum TRANS_TYPE : unsigned int{TRANS_UNINIT=0, DATA_TRANS=1, STATE_TRANS=2};


// 并行域入口{}形式的数据域, 并行域只在入口处可能存在数据传输操作 alloc to tofrom delete from 这5种类型可以描述
// 配对的数据域中的传输操作, 串行域中任何地方都可能出现数据传输

class TRANS{ // 用来描述数据传输、同步等操作
public:
    int index; // 变量在 vectorVar 中的 index
    TRANS_TYPE type;
    STATE_CONSTR StConstrTarget; // 表示要达到的 同步状态约束
    SourceLocation InsertLocation;
    bool isWritten; // 表示是否已经被写入源代码
    
    TRANS(int in_index, STATE_CONSTR in_StConstrTarget){
        index = in_index;
        StConstrTarget = in_StConstrTarget;
    }
        // : index(in_index), type(in_type) { isWritten=false; }
    
    TRANS(int in_index, TRANS_TYPE in_type, STATE_CONSTR in_StConstrTarget, SourceLocation in_Loc, Rewriter& Rewrite){
        index = in_index;
        type = in_type;
        StConstrTarget = in_StConstrTarget;
        InsertLocation = Rewrite.getSourceMgr().getExpansionLoc(in_Loc);
        isWritten=false;
    }
};

class CODE_REPLACE{
public:
    SourceRange Range;
    std::string Code;

    CODE_REPLACE(SourceRange in_Range, std::string in_Code, Rewriter& Rewrite){
        Range.setBegin( Rewrite.getSourceMgr().getExpansionLoc( in_Range.getBegin() ) );
        Range.setEnd( Rewrite.getSourceMgr().getExpansionLoc( in_Range.getEnd() ) );
        Code = in_Code;
    }
};

class CODE_INSERT{
public:
    SourceLocation InsertLoc;
    std::string Code;

    CODE_INSERT(SourceLocation in_InsertLoc, std::string in_Code, Rewriter& Rewrite){
        InsertLoc = Rewrite.getSourceMgr().getExpansionLoc(in_InsertLoc);
        Code = in_Code;
    }

    // 重载 ==
    bool operator==(const CODE_INSERT& in_CodeInsert){
        // 判断是否是同一个变量
        if(InsertLoc==in_CodeInsert.InsertLoc)
        {
            return true;
        }
        return false;
    }

    bool operator==(const SourceLocation& in_InsertLoc){
        // 判断是否是同一个变量
        if(InsertLoc==in_InsertLoc)
        {
            return true;
        }
        return false;
    }

    const CODE_INSERT& operator=(const CODE_INSERT& in_Insert){
        InsertLoc = in_Insert.InsertLoc;
        Code = in_Insert.Code;

        return in_Insert;
    }

    CODE_INSERT(const CODE_INSERT& in_Insert){
        InsertLoc = in_Insert.InsertLoc;
        Code = in_Insert.Code;
    }
};

class NODE_INDEX{ // 用来描述指向其他节点的 index , 相当于指针了
public:
    NODE_TYPE type; // 节点类型
    int  index; // 节点在对应 vector 中的 index

    bool operator == (const NODE_INDEX& in){
        if(type==in.type && index==in.index){
            return true;
        }
        return false;
    }

    bool operator != (const NODE_INDEX& in){
        if(type!=in.type || index!=in.index){
            return true;
        }
        return false;
    }
    
    NODE_INDEX& operator = (const NODE_INDEX& in){
        if(this != &in){
            type = in.type;
            index = in.index;
        }
        return *this;
    }

    NODE_INDEX(const NODE_INDEX& in) \
        : type(in.type), index(in.index) {}

    NODE_INDEX(NODE_TYPE in_type = NODE_TYPE::NODE_UNINIT, int in_index = -1) \
        : type(in_type), index(in_index) {}

    void init(NODE_TYPE in_type, int in_index){
        type = in_type;
        index = in_index;
    }
    void init(){
        type = NODE_TYPE::NODE_UNINIT;
        index = -1;
    }
};

// 描述某个变量的一次引用类型
class VAR_REF{
public:
    enum MEMBER_TYPE{ NO_MEMBER=-1, DOT=0, ARROW=1 }; // 表示成员运算符的类型
    MEMBER_TYPE MemberType; // 这个在变量是类的域时才有意义, 1表示使用“->”来引用类的域, 0表示使用“.”来引用类的域, -1表示不是类的域
    // REF_TYPE RefType; // 引用的读写类型
    MY_SOURCE_RANGE SrcRange; // 引用在源码中的位置信息, 这个必须获得, 因为函数引用的实参要靠后处理, 插入引用信息时需要判断插入位置

    // wfr 20190319 这两个好像用不到了
    // 如果该引用是作为函数的参数
    int indexFunc; // 表示函数的 index
    int indexParm; // 表示是函数的第几个参数

    STATE_CONSTR StReq, StTransFunc; // 入口同步状态需求, 同步状态转换函数
    
    VAR_REF(MEMBER_TYPE in_MemberType, STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc, MY_SOURCE_RANGE& in_SrcRange){
        init(in_MemberType, in_StReq, in_StTransFunc, in_SrcRange);
    }
    const VAR_REF& operator=(const VAR_REF& in){
        init(in);
        return in;
    }
    VAR_REF(const VAR_REF& in){
        init(in);
    }
    VAR_REF(){
        init();
    }
    void init(const VAR_REF& in){
        MemberType = in.MemberType;
        SrcRange = in.SrcRange;
        indexFunc = in.indexFunc;
        indexParm = in.indexParm;
        StReq = in.StReq;
        StTransFunc = in.StTransFunc;
    }
    void init(MEMBER_TYPE in_MemberType, STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc, MY_SOURCE_RANGE& in_SrcRange){
        MemberType = in_MemberType; SrcRange = in_SrcRange;
        indexFunc = -1; indexParm = -1;
        StReq = in_StReq;
        StTransFunc = in_StTransFunc;
    }
    void init(){
        MemberType = MEMBER_TYPE::NO_MEMBER; // 这个在变量是类的域时才有意义, 1表示使用“->”来引用类的域, 0表示使用“.”来引用类的域, -1表示不是类的域
        indexFunc = -1; // 表示函数的 index
        indexParm = -1;
    }
};

// 描述变量引用类型
class VAR_REF_LIST{
public:
    int index; // 变量在 vectorVar 中的 index

    SYNC_STATE SyncStateBegin; // 节点开始时的对于变量的同步状态的要求
    SYNC_STATE SyncStateEnd; // 节点结束后参数变量的同步状态

    STATE_CONSTR StReq, StTransFunc; // 入口同步状态需求, 同步状态转换函数
    bool NeedInsertStTrans; // wfr 20190724 标识该节点之后是否需要插入状态转移函数

    std::vector<VAR_REF> RefList;

    // 如果是 OMP 中的函数引用, 保存以下三种信息即可, 等到在 AnalyzeOneFunction 中再进一步处理
    int indexCallee; // 被调函数的 index
    CallExpr* pFuncCall; // 调用语句的 指针
    std::vector<int> vectorArgIndex; // 各个实参的 index, 是按顺序的

    VAR_REF_LIST(int in_index, SYNC_STATE in_SyncStateBegin, SYNC_STATE in_SyncStateEnd, 
                STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc)
    {
        init(in_index, in_SyncStateBegin, in_SyncStateEnd, in_StReq, in_StTransFunc);
    }
    void init(int in_index, SYNC_STATE in_SyncStateBegin, SYNC_STATE in_SyncStateEnd, 
              STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc){
        index = in_index; SyncStateBegin = in_SyncStateBegin; SyncStateEnd = in_SyncStateEnd;
        indexCallee=-1; pFuncCall=NULL;
        StReq = in_StReq;
        StTransFunc = in_StTransFunc;
        NeedInsertStTrans = true;
    }

    VAR_REF_LIST(int in_index = -1){
        init(in_index);
    }
    void init(int in_index = -1) {
        index = in_index; indexCallee=-1; pFuncCall=NULL;
        SyncStateBegin = SYNC_STATE::SYNC_UNINIT;
        SyncStateEnd = SYNC_STATE::SYNC_UNINIT;
        NeedInsertStTrans = true;
    }

    // 重载 == , 用 find 函数在 vector 里 按照 indexVar 搜索时, 会用到这个
    bool operator==(int in_index){
        // 判断是否是同一个变量
        if(index==in_index)
        {
            return true;
        }
        return false;
    }
    
};

class BLOCK_INFO{ // 阻塞只涉及上下两层节点, 上层父节点之间需要协商 出口同步状态
public:
    std::vector<NODE_INDEX> Parents;
    std::vector<NODE_INDEX> Children;
};

class SEQ_PAR_NODE_BASE{
public:
    std::vector<NODE_INDEX> vectorParents; // 父节点 index 的 vector
    std::vector<NODE_INDEX> vectorChildren; // 子节点 index 的 vector
    std::vector<VAR_REF_LIST> vectorVarRef;
    //vector<SYNC_STATE> vectorSyncBegin; // 节点开始时对同步状态的要求, 每个变量对应一个表项
    //vector<SYNC_STATE> vectorSyncEnd; // 节点执行完时的同步状态 vector, 每个变量对应一个表项
    std::vector<TRANS> vectorTrans; // 需要加入的数据传输/同步操作列表: 插入代码的位置
    //Stmt* pCursor;
    bool isComplete; // 表示该节点是否处理完成
    //bool OMPParallel; //该节点是否为pragma所在节点，若是，需要将子节点并入
    std::string JumpStmt; // 跳转指令的名称字符串
    Stmt* TerminatorStmt; // 跳转指令的地址
    MY_SOURCE_RANGE ReturnRange; // 如果该节点中有 return 语句, 将范围保存在这里, 因为一个节点中的语句是顺序的没有分支, 因此至多只能有一个 return 语句 且 一定是该节点最后一句
    MY_SOURCE_RANGE LoopBodyRange; // 循环体范围
    MY_SOURCE_RANGE ThenRange; // then body 的范围
    MY_SOURCE_RANGE ElseRange; // else body 的范围
    NODE_INDEX LoopBody; // wfr 20190725 表示 循环体 的 第一个节点
    NODE_INDEX LoopExit; //  wfr 20190725 表示 循环之后 的 第一个节点
    bool LoopNeedAnalysis;

    // 以下信息是遍历串并行图的时候用到的
    int indexVar; // 变量在 vectorVar 中的 index
    TRANS_TYPE EntryOperation; // 入口同步操作
    SYNC_STATE EntryState; // 入口同步后的入口状态（即满足入口要求, 只是进一步确认是不是SYNC）
    TRANS_TYPE ExitOperation; // 出口同步操作
    SYNC_STATE ExitState; // 协商出口状态
    bool isBlocked; // 是否处于阻塞状态
    //bool isInDevice; // 标识变量是否 map 到 device 上
    bool isNegotiated; // 标识 该节点 是否经过了协商
    bool isProcessed;
    bool isMapped; // 标识变量是否被 map 了, 有时即使变量在 device 中, 也需要 alloc 来建立关联
    bool NeedInsertStTrans; // wfr 20190724 标识该节点之后是否需要插入状态转移函数

    STATE_CONSTR EntryStTrans; // 入口所需的同步状态转换函数
    STATE_CONSTR ExitStTrans; // 出口所需的同步状态转换函数
    STATE_CONSTR EntryStConstr; // 入口处同步状态满足的约束
    STATE_CONSTR ExitStConstr; // 出口处同步状态满足的约束

    // 初始化 遍历 串并行图 所需的信息
    void initSYNCInfo(int in_index){
        if(in_index!=indexVar){
            resetSYNCInfo(in_index);
        }
    }

    // 重置 遍历 串并行图 所需的信息
    void resetSYNCInfo(int in_index){
        indexVar = in_index;
        EntryOperation = TRANS_TYPE::TRANS_UNINIT;
        ExitOperation = TRANS_TYPE::TRANS_UNINIT;
        EntryState = SYNC_STATE::SYNC_UNINIT;
        ExitState = SYNC_STATE::SYNC_UNINIT;
        isBlocked = false;
        //isInDevice = false;
        isNegotiated = false;
        isProcessed = false;
        isMapped = false;
        NeedInsertStTrans = true;
        LoopNeedAnalysis = false;

        EntryStTrans.init(ST_TRANS_UNINIT);
        ExitStTrans.init(ST_TRANS_UNINIT);
        EntryStConstr.init(ST_REQ_UNINIT);
        ExitStConstr.init(ST_REQ_UNINIT);
    }

    void init(const SEQ_PAR_NODE_BASE& in_Base){
        vectorParents = in_Base.vectorParents;
        vectorChildren = in_Base.vectorChildren;
        vectorVarRef = in_Base.vectorVarRef;
        vectorTrans = in_Base.vectorTrans;
        isComplete = in_Base.isComplete;
        JumpStmt = in_Base.JumpStmt;
        TerminatorStmt = in_Base.TerminatorStmt;
        ReturnRange = in_Base.ReturnRange;
        LoopBodyRange = in_Base.LoopBodyRange;
        ThenRange = in_Base.ThenRange;
        ElseRange = in_Base.ElseRange;

        indexVar = in_Base.indexVar;
        EntryOperation = in_Base.EntryOperation;
        EntryState = in_Base.EntryState;
        ExitOperation = in_Base.ExitOperation;
        ExitState = in_Base.ExitState;
        isBlocked = in_Base.isBlocked;
        isNegotiated = in_Base.isNegotiated;
        isProcessed = in_Base.isProcessed;
        isMapped = in_Base.isMapped;
        NeedInsertStTrans = in_Base.NeedInsertStTrans;
        LoopNeedAnalysis = in_Base.LoopNeedAnalysis;

        EntryStTrans = in_Base.ExitStTrans;
        ExitStTrans = in_Base.ExitStTrans;
        EntryStConstr = in_Base.EntryStConstr;
        ExitStConstr = in_Base.ExitStConstr;
    }

    void init(){
        isComplete = false;
        JumpStmt = "NULL";
        TerminatorStmt = NULL;
        indexVar = -1;
        EntryOperation = TRANS_TYPE::TRANS_UNINIT;
        ExitOperation = TRANS_TYPE::TRANS_UNINIT;
        EntryState = SYNC_STATE::SYNC_UNINIT;
        ExitState = SYNC_STATE::SYNC_UNINIT;
        isBlocked = false;
        isNegotiated = false;
        isProcessed = false;
        isMapped = false;
        NeedInsertStTrans = true;
        LoopNeedAnalysis = false;

        EntryStTrans.init(ST_TRANS_UNINIT);
        ExitStTrans.init(ST_TRANS_UNINIT);
        EntryStConstr.init(ST_REQ_UNINIT);
        ExitStConstr.init(ST_REQ_UNINIT);
    }

    const SEQ_PAR_NODE_BASE& operator=(const SEQ_PAR_NODE_BASE& in_Base){
        init(in_Base);
        return in_Base;
    }

    SEQ_PAR_NODE_BASE(const SEQ_PAR_NODE_BASE& in_Base){
        init(in_Base);
    }

    SEQ_PAR_NODE_BASE(){
        init();
    }
};

#endif // SEQ_PAR_GRAPH_BASE_H
