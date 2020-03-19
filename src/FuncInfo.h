/***************************************************************************
Copyright(C), 2010-2018, 瑞雪轻飏
     FileName: FuncInfo.h
       Author: 瑞雪轻飏
      Version: 0.1
Creation Date: 20190130
  Description: 函数信息类, 一个函数是一个串并行域图
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

#ifndef FUNC_INFO_H
#define FUNC_INFO_H

#include "Clang_LLVM_Inc.h"
#include "OMPRegion.h"
#include "SequentialRegion.h"
#include "SrcRange.h"
#include "VarInfo.h"

using namespace clang;

// wfr 20191225 从 std::vector<NODE_INDEX> 中 删除 NODE_INDEX
void deleteNodeIndex(std::vector<NODE_INDEX> &vectorNodeIndex, NODE_INDEX indexNode);
// wfr 20191225 删除节点指向自身的边
void deleteEdge2Self(NODE_INDEX indexSEQ, std::vector<SEQ_REGION> &vectorSEQ);
void deleteEdge2Self(NODE_INDEX indexOMP, std::vector<OMP_REGION> &vectorOMP);
// wfr 20191225 合并节点, 将 src 节点 合并到 dest 节点, 只将 src 节点空间重置 并不释放
void merge(NODE_INDEX indexDest, NODE_INDEX indexSrc, std::vector<SEQ_REGION> &vectorSEQ, std::vector<OMP_REGION> &vectorOMP);
// 合并 SEQ 节点, 将 src 节点 合并到 dest 节点, 只将 src 节点空间重置 并不释放
void merge(int dest, int src, std::vector<SEQ_REGION> &vectorSEQ, Rewriter& Rewrite);

// 函数输入参数列表: 该参数的类型(拷贝/指针/引用), 该参数对同步状态的要求, 函数结束时该参数的同步状态信息
// 如果传递的是普通变量的副本, 则不存在写操作
// 传递 指针 / 引用 才可能存在写操作
// 对于 类 / 结构体 要对引用的域分别讨论
// 利用 const 方便分析

// 对于类和类中的域, 类定义打头, 后边跟着类中各个被应用到的域
// 函数的参数也是局部变量, 也应该放入函数的局部变量列表 vectorLocal 中
class FUNC_PARM_VAR_INFO {
public:
    // std::string    Name; // 变量名/域的名字
    // Decl*          ptrDecl; // 变量的定义的地址
    // std::string    TypeName; // 变量的类型：int、int*、float*等等
    // bool           isMember; // 1表示变量是类的域, 0表示不是, 默认是0
    // bool           isArrow; // 这个在变量是类的域时才有意义, 1表示使用“->”来引用类的域, 0表示使用“.”来引用类的域
    // Decl*          ptrClass; // 如果是类的成员, 这里是类实例定义的地址
    // unsigned int   indexClass; // 如果是类的成员, 这里是类的 VARIABLE_INFO 在列表中的 index

    int indexVar; // 变量在 vectorLocal 中的 index
    // int     indexClassParm; // 指向 当前域 所属的 类 在参数列表中 的 index, 不是域就指向当前节点自身
    bool UsedInOMP; // 标识该变量是否在 OMP 中被调用

    // 这里应该加上变量类型信息
    // 这里应该加上变量类型信息
    // 1. 传入拷贝(且不是指针)(不改变)
    // 2. 简单指针(可以有const修饰)(可能改变)
    // 3. const TYPE* [const] VAR(认为不改变)
    // 4. 简单引用/类的引用(可能改变)
    // 5. 类的指针(可以有const修饰)(可能改变)
    // 6. const CLASS_TYPE* [const] VAR(认为不改变)
    // 7. 类(认为可能改变, 因为类的域可能是指针变量)
    // enum PARM_TYPE {PARM_UNINIT, COPY, PTR, PTR_CONST, CLASS_PTR, CLASS_PTR_CONST, REF, CLASS_REF, CLASS};
    VAR_TYPE ParmType;

    // 同步状态要求, 结束时同步状态: 同步, host最新, device最新, host-only, 无要求, 不改变
    SYNC_STATE SyncStateBegin; // 对于函数开始时的参数变量的同步状态的要求
    SYNC_STATE SyncStateEnd;   // 函数结束后参数变量的同步状态
    STATE_CONSTR StReq, StTransFunc; // 入口同步状态需求, 同步状态转换函数
    bool NeedInsertStTrans; // wfr 20190724 标识该节点之后是否需要插入状态转移函数

    FUNC_PARM_VAR_INFO(int in_indexVar, VAR_TYPE in_ParmType,
                       SYNC_STATE in_SyncStateBegin, SYNC_STATE in_SyncStateEnd, 
                       STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc) {
        init(in_indexVar, in_ParmType, in_SyncStateBegin, in_SyncStateEnd, in_StReq, in_StTransFunc);
    }

    FUNC_PARM_VAR_INFO(int in_indexVar, VAR_TYPE in_ParmType){
        init(in_indexVar, in_ParmType);
    }

    FUNC_PARM_VAR_INFO(int in_indexVar){
        init(in_indexVar);
    }

    FUNC_PARM_VAR_INFO(const FUNC_PARM_VAR_INFO& in){
        init(in);
    }

    const FUNC_PARM_VAR_INFO& operator=(const FUNC_PARM_VAR_INFO& in){
        init(in);
        return in;
    }

    void init(const FUNC_PARM_VAR_INFO& in){
        indexVar = in.indexVar;
        UsedInOMP = in.UsedInOMP;
        ParmType = in.ParmType;
        SyncStateBegin = in.SyncStateBegin;
        SyncStateEnd = in.SyncStateEnd;
        StReq = in.StReq;
        StTransFunc = in.StTransFunc;
        NeedInsertStTrans = in.NeedInsertStTrans;
    }

    void init(int in_indexVar, VAR_TYPE in_ParmType,
              SYNC_STATE in_SyncStateBegin, SYNC_STATE in_SyncStateEnd, 
              STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc) {
        indexVar = in_indexVar;
        ParmType = in_ParmType;
        SyncStateBegin = in_SyncStateBegin;
        SyncStateEnd = in_SyncStateEnd;
        StReq = in_StReq;
        StTransFunc = in_StTransFunc;
        UsedInOMP = false;
        NeedInsertStTrans = true;
    }

    void init(int in_indexVar, VAR_TYPE in_ParmType){
        indexVar = in_indexVar;
        ParmType = in_ParmType;
        SyncStateBegin = SYNC_STATE::SYNC_UNINIT;
        SyncStateEnd = SYNC_STATE::SYNC_UNINIT;
        UsedInOMP = false;
        NeedInsertStTrans = true;
    }

    void init(int in_indexVar){
        indexVar = in_indexVar;
        ParmType = VAR_TYPE::VAR_UNINIT;
        SyncStateBegin = SYNC_STATE::SYNC_UNINIT;
        SyncStateEnd = SYNC_STATE::SYNC_UNINIT;
        UsedInOMP = false;
        NeedInsertStTrans = true;
    }

    // 重载 == , 用 find 函数在 vector 里 按照 indexVar 搜索时, 会用到这个
    bool operator==(int in_index) {
        // 判断是否是同一个变量
        if (indexVar == in_index) {
            return true;
        }
        return false;
    }
};

class FUNC_INFO : public MY_SOURCE_RANGE {
public:
    std::string Name;
    FunctionDecl *ptrDecl;   // 函数的声明的地址
    FunctionDecl *ptrDefine; // 定义地址, 有时是先声明后定义
    bool isComplete;         // 表示函数信息是否保存完整了
    bool UsedInOMP;          // 表示该函数是否在并行域中被调用
    int NumParams;           // 函数参数个数

    FunctionTemplateDecl *ptrTemplate; // 函数模版的地址

    MY_SOURCE_RANGE CompoundRange; // 保存函数的 {} 的范围

    // 串并行域图的第一个节点应该是串行节点, 存在该 vector 的第一个, 同样最后一个也应该是串行节点, 存在该 vector
    // 的最后一个
    std::vector<SEQ_REGION> vectorSEQ; // 该函数中串行节点的vector
    std::vector<OMP_REGION> vectorOMP; // 该函数中并行节点的vector
    std::vector<VARIABLE_INFO> vectorVar; // 变量列表, 记录函数中的并行域中使用到的, 定义在并行域外的变量
    // std::vector<VARIABLE_INFO>   vectorLocal; // 局部变量列表, 定义在并行域内的变量, 每次遇到新并行域就清空,
    // 这个应该定义到处理OMP域的函数里
    std::vector<FUNC_PARM_VAR_INFO> vectorParm; // 保存函数参数独有的信息
    // 函数输入参数列表: 该参数的类型(拷贝/指针/引用), 该参数对同步状态的要求, 函数结束时该参数的同步状态信息
    // 要处理参数是类/结构体的情况, 使用了类/结构体中的哪些域, 将类/结构体的域当作单独的变量来分析
    std::vector<int> vectorCallee; // wfr 20190724 保存当前函数调用的函数的 index

    std::vector<VARIABLE_INFO> vectorDeclVar;       // 变量声明时的变量列表
    std::vector<FUNC_PARM_VAR_INFO> vectorDeclParm; // 变量声明时的参数列表

    std::vector<CODE_REPLACE> vectorReplace; // wfr 20190720 保存对源代码的替换操作, 主要是替换 malloc 和 free

    // 串并行图的 入口/出口 节点都应该是串行节点
    NODE_INDEX MapEntry; // 保存串并行图的入口节点的 index
    NODE_INDEX MapExit;  // 保存串并行图的出口节点的 index

    // ~FUNC_INFO(){
    //     std::cout << "~FUNC_INFO: 析构 函数" << this->Name << " 的信息" << std::endl;
    //     std::cout << "~FUNC_INFO: vectorParm.size() = " << vectorParm.size() << std::endl;  
    // }

    FUNC_INFO(const FUNC_INFO& in){
        init(in);
    }

    const FUNC_INFO& operator=(const FUNC_INFO& in){
        init(in);
        return in;
    }

    void init(const FUNC_INFO& in){

        (MY_SOURCE_RANGE&)(*this) = (MY_SOURCE_RANGE&)in;

        Name = in.Name;
        ptrDecl = in.ptrDecl;
        ptrDefine = in.ptrDefine;
        isComplete = in.isComplete;
        UsedInOMP = in.UsedInOMP;
        NumParams = in.NumParams;
        ptrTemplate = in.ptrTemplate;
        CompoundRange = in.CompoundRange;
        vectorSEQ = in.vectorSEQ;
        vectorOMP = in.vectorOMP;
        vectorVar = in.vectorVar;
        vectorParm = in.vectorParm;
        vectorCallee = in.vectorCallee;
        vectorDeclVar = in.vectorDeclVar;
        vectorDeclParm = in.vectorDeclParm;
        vectorReplace = in.vectorReplace;
        MapEntry = in.MapEntry;
        MapExit = in.MapExit;
    }

    void init(Rewriter &Rewrite, CFG &FuncCFG) {

        std::cout << "FUNC_INFO::init 使用 CFG 初始化 函数" << std::endl;

        int NumBlocks = FuncCFG.getNumBlockIDs();
        vectorSEQ.resize(NumBlocks);

        // 遍历 CFG, 建立串并行图最初的框架
        // 最初始的串并行图中节点的 index 与 CFG中节点的 BlockID 相同, 这样节点间边的关系才可以直接映射过来
        // 初始化entry节点
        int indexBlock;
        std::string StmtClassName;
        CFGBlock &EntryBlock = FuncCFG.getEntry();
        indexBlock = EntryBlock.getBlockID();
        MapEntry.init(NODE_TYPE::SEQUENTIAL, indexBlock);
        vectorSEQ[indexBlock].init(EntryBlock, Rewrite);

        int flag[NumBlocks]; //0代表seq节点，1代表omp节点，-1代表已经被合并
        for(int i=0; i<NumBlocks; i++){
            flag[i] = 0;
        }

        // 循环初始化其他节点
        // Iterate through the CFGBlocks and print them one by one.
        for (clang::CFG::iterator I = FuncCFG.begin(); I != FuncCFG.end(); ++I) {
            // Skip the entry block, because we already printed it.
            if (&(**I) == &FuncCFG.getEntry() || &(**I) == &FuncCFG.getExit())
                continue;

            if (*I != NULL) {
                indexBlock = (**I).getBlockID();                
                vectorSEQ[indexBlock].init((**I), Rewrite);
                //ghn 20191105 判断OMP语句（omp节点在clang9的CFG图与之前不一样，需要重新判断）
                for(CFGBlock::const_iterator iter=(**I).begin(); iter!=(**I).end(); ++iter){
                    const Stmt* pStmt; pStmt = NULL;
                    pStmt = dyn_cast<Stmt>( (*iter).castAs<CFGStmt>().getStmt() );
                    std::cout << "----节点类型名称 " << pStmt->getStmtClassName() << std::endl;
                    if(strcmp(pStmt->getStmtClassName(),"OMPParallelForDirective") == 0){
                        flag[indexBlock] = 1;
                    }
                    std::cout << "----flag :" << flag[indexBlock] << std::endl;
                }
                
                //StmtClassName = (**I).getCFGElementStmtPtr()->getStmtClassName();
                // 获得跳转指令的名称字符串
                if ((**I).getTerminatorStmt()) {
                    vectorSEQ[indexBlock].TerminatorStmt = (**I).getTerminatorStmt();
                    vectorSEQ[indexBlock].JumpStmt = (**I).getTerminatorStmt()->getStmtClassName();
                    //std::cout << "节点类型名称: " << vectorSEQ[indexBlock].JumpStmt << std::endl;
                }
            }
        }

        // 初始化exit节点
        CFGBlock &ExitBlock = FuncCFG.getExit();
        indexBlock = ExitBlock.getBlockID();
        MapExit.init(NODE_TYPE::SEQUENTIAL, indexBlock);
        vectorSEQ[indexBlock].init(ExitBlock, Rewrite);

        // wfr 20190505 在这里 merge while/for/if 相关节点
        for (unsigned long i = 0; i < vectorSEQ.size(); ++i) {
            NODE_INDEX indexSEQ;
            indexSEQ.init(NODE_TYPE::SEQUENTIAL, i);
            SEQ_REGION &SEQ = vectorSEQ[i];
            //ghn 20191105 加入flag判断
            if ( SEQ.vectorParents.empty() && SEQ.vectorChildren.empty() ) {
                continue;
            }

            if (SEQ.TerminatorStmt != NULL && isa<IfStmt>(SEQ.TerminatorStmt)) {
                IfStmt *pIfStmt = static_cast<IfStmt *>(SEQ.TerminatorStmt);
                // Stmt *pFirstChild = *(pIfStmt->children().begin());

                // 设置 if 语句(到判断条件结束) 的范围
                SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pIfStmt->getBeginLoc());
                if (TmpLoc < SEQ.SEQRange.BeginLoc) {
                    SEQ.SEQRange.SetBeginLoc(TmpLoc, Rewrite);
                }
                TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pIfStmt->getThen()->getBeginLoc());
                if (SEQ.SEQRange.EndLoc < TmpLoc) {
                    SEQ.SEQRange.SetEndLoc(TmpLoc, Rewrite);
                }

                // wfr 20190509 保存 than 范围
                SEQ.ThenRange.init(pIfStmt->getThen()->getBeginLoc(), pIfStmt->getThen()->getEndLoc(), Rewrite);
                // wfr 20190815 如果结尾不是 "}", 就对结尾位置进行修复
                TmpLoc = SEQ.ThenRange.EndLoc;
                FileID Fileid = Rewrite.getSourceMgr().getFileID(TmpLoc);
                // StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
                const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
                const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
                const char* start = buffer->getBufferStart();
                if(start[SEQ.ThenRange.EndOffset]!='}'){
                    SEQ.ThenRange.SetEndLoc(fixStmtLoc(SEQ.ThenRange.EndLoc, Rewrite), Rewrite);
                }

                // wfr 20190509 保存 else 范围
                if(pIfStmt->getElse()!=NULL){
                    SEQ.ElseRange.init(pIfStmt->getElse()->getBeginLoc(), pIfStmt->getElse()->getEndLoc(), Rewrite);
                    // wfr 20190815 如果结尾不是 "}", 就对结尾位置进行修复
                    TmpLoc = SEQ.ElseRange.EndLoc;
                    FileID Fileid = Rewrite.getSourceMgr().getFileID(TmpLoc);
                    // StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
                    const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
                    const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
                    const char* start = buffer->getBufferStart();
                    if(start[SEQ.ElseRange.EndOffset]!='}'){
                        SEQ.ElseRange.SetEndLoc(fixStmtLoc(SEQ.ElseRange.EndLoc, Rewrite), Rewrite);
                    }
                }

                // merge 属于 if 语句 的其他节点
                SEQ_REGION *pSEQParent = NULL;
                SourceLocation EmptyLoc;
                NODE_INDEX indexParent; // 回溯到的整个 if() 结构的入口节点
                for (unsigned long j = 0; j < SEQ.vectorParents.size(); ++j) {
                    indexParent = SEQ.vectorParents[j];
                    pSEQParent = &vectorSEQ[indexParent.index];
                    if(indexParent==indexSEQ){
                        continue;
                    }else if (SEQ.SEQRange.BeginOffset <= pSEQParent->SEQRange.BeginOffset &&
                        pSEQParent->SEQRange.EndOffset <= SEQ.SEQRange.EndOffset) {
                        merge(indexSEQ.index, indexParent.index, vectorSEQ, Rewrite);
                        j--;
                    } else if (pSEQParent->SEQRange.BeginLoc == EmptyLoc && pSEQParent->SEQRange.EndLoc == EmptyLoc &&
                               indexParent != MapEntry && indexParent != MapExit) {
                        merge(indexSEQ.index, indexParent.index, vectorSEQ, Rewrite);
                        j--;
                    } else {
                    }
                }
            } else if (SEQ.TerminatorStmt != NULL && isa<WhileStmt>(SEQ.TerminatorStmt)) {
                WhileStmt *pWhileStmt = static_cast<WhileStmt *>(SEQ.TerminatorStmt);
                // Stmt *pFirstChild = *(pWhileStmt->children().begin());

                // 设置 while 语句(到判断条件结束) 的范围
                SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pWhileStmt->getBeginLoc());
                if (TmpLoc < SEQ.SEQRange.BeginLoc) {
                    SEQ.SEQRange.SetBeginLoc(TmpLoc, Rewrite);
                }
                TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pWhileStmt->getBody()->getBeginLoc());
                if (SEQ.SEQRange.EndLoc < TmpLoc) {
                    SEQ.SEQRange.SetEndLoc(TmpLoc, Rewrite);
                }

                // wfr 20190509 保存 循环体 范围
                SEQ.LoopBodyRange.init(pWhileStmt->getBody()->getBeginLoc(), pWhileStmt->getBody()->getEndLoc(), Rewrite);

                // wfr 20190815 如果循环体结尾不是 "}", 就对结尾位置进行修复
                TmpLoc = SEQ.LoopBodyRange.EndLoc;
                FileID Fileid = Rewrite.getSourceMgr().getFileID(TmpLoc);
                // StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
                const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
                const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
                const char* start = buffer->getBufferStart();
                if(start[SEQ.LoopBodyRange.EndOffset]!='}'){
                    SEQ.LoopBodyRange.SetEndLoc(fixStmtLoc(SEQ.LoopBodyRange.EndLoc, Rewrite), Rewrite);
                }

                // 重置 属于 while 语句 的其他节点
                SEQ_REGION *pSEQParent = NULL;
                SourceLocation EmptyLoc;
                NODE_INDEX indexParent; // 回溯到的整个 while() 结构的入口节点
                for (unsigned long j = 0; j < SEQ.vectorParents.size(); ++j) {
                    indexParent = SEQ.vectorParents[j];
                    pSEQParent = &vectorSEQ[indexParent.index];
                    if(indexParent==indexSEQ){
                        continue;
                    }else if (SEQ.SEQRange.BeginOffset <= pSEQParent->SEQRange.BeginOffset &&
                        pSEQParent->SEQRange.EndOffset <= SEQ.SEQRange.EndOffset) {
                        merge(indexSEQ.index, indexParent.index, vectorSEQ, Rewrite);
                        j--;
                    } else if (pSEQParent->SEQRange.BeginLoc == EmptyLoc && pSEQParent->SEQRange.EndLoc == EmptyLoc &&
                               indexParent != MapEntry && indexParent != MapExit) {
                        merge(indexSEQ.index, indexParent.index, vectorSEQ, Rewrite);
                        j--;
                    } else {
                    }
                }
            } else if (SEQ.TerminatorStmt != NULL && isa<ForStmt>(SEQ.TerminatorStmt)) {
                ForStmt *pForStmt = static_cast<ForStmt *>(SEQ.TerminatorStmt);

                // 设置 for 语句(到判断条件结束) 的范围
                SourceLocation TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pForStmt->getBeginLoc());
                if (TmpLoc < SEQ.SEQRange.BeginLoc) {
                    SEQ.SEQRange.SetBeginLoc(TmpLoc, Rewrite);
                }
                TmpLoc = Rewrite.getSourceMgr().getExpansionLoc(pForStmt->getBody()->getBeginLoc());
                if (SEQ.SEQRange.EndLoc < TmpLoc) {
                    SEQ.SEQRange.SetEndLoc(TmpLoc, Rewrite);
                }

                // wfr 20190509 保存 循环体 范围
                SEQ.LoopBodyRange.init(pForStmt->getBody()->getBeginLoc(), pForStmt->getBody()->getEndLoc(), Rewrite);

                // wfr 20190815 如果循环体结尾不是 "}", 就对结尾位置进行修复
                TmpLoc = SEQ.LoopBodyRange.EndLoc;
                FileID Fileid = Rewrite.getSourceMgr().getFileID(TmpLoc);
                // StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
                const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
                const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
                const char* start = buffer->getBufferStart();
                if(start[SEQ.LoopBodyRange.EndOffset]!='}'){
                    SEQ.LoopBodyRange.SetEndLoc(fixStmtLoc(SEQ.LoopBodyRange.EndLoc, Rewrite), Rewrite);
                }

                // 重置 属于 for 语句 的其他节点
                SEQ_REGION *pSEQParent = NULL;
                SEQ_REGION *pSEQChild = &SEQ;
                SourceLocation EmptyLoc;
                NODE_INDEX indexParent; // 回溯到的整个 for() 结构的入口节点
                for (unsigned long j = 0; j < SEQ.vectorParents.size(); ++j) {
                    indexParent = SEQ.vectorParents[j];
                    pSEQParent = &vectorSEQ[indexParent.index];
                    if(indexParent==indexSEQ){
                        continue;
                    }else if (SEQ.SEQRange.BeginOffset <= pSEQParent->SEQRange.BeginOffset &&
                        pSEQParent->SEQRange.EndOffset <= SEQ.SEQRange.EndOffset) {
                        merge(indexSEQ.index, indexParent.index, vectorSEQ, Rewrite);
                        j--;
                    } else if (pSEQParent->SEQRange.BeginLoc == EmptyLoc && pSEQParent->SEQRange.EndLoc == EmptyLoc &&
                               indexParent != MapEntry && indexParent != MapExit) {
                        merge(indexSEQ.index, indexParent.index, vectorSEQ, Rewrite);
                        j--;
                    } else {
                    }
                }
            } else if (flag[i] == 1) {
                //ghn 20191105 设置OMP节点与子节点融合
                
                // 重置 属于 OMPParallelForDirective 语句 的其他节点
                SEQ_REGION *pSEQChild = NULL;
                //ghn 20191105 子节点begin < = > pragma节点，但是结束一定在pragma节点之 内
                for (unsigned long j = 0; j < vectorSEQ.size(); ++j) {
                    pSEQChild = &vectorSEQ[j];
                    if(i==j){ //本身
                        continue;
                    }else if (SEQ.SEQRange.BeginOffset <= pSEQChild->SEQRange.EndOffset &&
                        pSEQChild->SEQRange.EndOffset <= SEQ.SEQRange.EndOffset) { //EndLoc在omp节点内
                        merge(indexSEQ.index, j, vectorSEQ, Rewrite);
                        flag[j] = -1; //被合并，置为-1
                        j--;
                    }else {
                    }
                }

            } else {
            }
        }

    }

    void init(FunctionDecl *in_ptrDecl, FunctionDecl *in_ptrDefine, Rewriter &Rewrite, CFG &FuncCFG) {
        isComplete = false;
        UsedInOMP = false;
        ptrDecl = in_ptrDecl;
        ptrDefine = in_ptrDefine;
        ptrTemplate = ptrDecl->getPrimaryTemplate();
        NumParams = in_ptrDecl->getNumParams();
        if (ptrDecl != NULL) {
            Name = ptrDecl->getNameAsString();
        } else {
            Name = "NULL";
        }
        init(Rewrite, FuncCFG);

        BeginLoc = Rewrite.getSourceMgr().getExpansionLoc(in_ptrDefine->getSourceRange().getBegin());
        EndLoc = Rewrite.getSourceMgr().getExpansionLoc(in_ptrDefine->getSourceRange().getEnd());
        BeginLine = Rewrite.getSourceMgr().getSpellingLineNumber(BeginLoc);
        BeginCol = Rewrite.getSourceMgr().getSpellingColumnNumber(BeginLoc);
        EndLine = Rewrite.getSourceMgr().getSpellingLineNumber(EndLoc);
        EndCol = Rewrite.getSourceMgr().getSpellingColumnNumber(EndLoc);
        BeginOffset = Rewrite.getSourceMgr().getFileOffset(BeginLoc);
        EndOffset = Rewrite.getSourceMgr().getFileOffset(EndLoc);
    }

    FUNC_INFO(FunctionDecl *in_ptrDecl, FunctionDecl *in_ptrDefine, Rewriter &Rewrite, CFG &FuncCFG) {
        init(in_ptrDecl, in_ptrDefine, Rewrite, FuncCFG);
    }
    
    FUNC_INFO(FunctionDecl *in_ptrDecl, Rewriter &Rewrite, CFG &FuncCFG)
      : MY_SOURCE_RANGE(in_ptrDecl->getBeginLoc(), in_ptrDecl->getEndLoc(), Rewrite) {
        isComplete = false;
        UsedInOMP = false;
        ptrDecl = in_ptrDecl;
        ptrTemplate = ptrDecl->getPrimaryTemplate();
        ptrDefine = NULL;
        if (ptrDecl != NULL) {
            Name = ptrDecl->getNameAsString();
        } else {
            Name = "NULL";
        }
        NumParams = in_ptrDecl->getNumParams();
        init(Rewrite, FuncCFG);
    }

    FUNC_INFO(FunctionDecl *in_ptrDecl) {
        isComplete = false;
        UsedInOMP = false;
        ptrDecl = in_ptrDecl;
        ptrTemplate = ptrDecl->getPrimaryTemplate();
        ptrDefine = NULL;
        if (ptrDecl != NULL) {
            Name = ptrDecl->getNameAsString();
        } else {
            Name = "NULL";
        }
        if(Name=="free"){
            NumParams = 1;
        }else{
            NumParams = in_ptrDecl->getNumParams();
        }
    }

    FUNC_INFO() : MY_SOURCE_RANGE() {
        isComplete = false;
        UsedInOMP = false;
        Name = "NULL";
        ptrDecl = NULL;
        ptrDefine = NULL;
        ptrTemplate = NULL;
        NumParams = -1;
    }
};

#endif // FUNC_INFO_H
