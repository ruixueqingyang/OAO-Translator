/*******************************************************************************
Copyright(C), 2010-2019, 瑞雪轻飏
     FileName: OAORewriter.h
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190105
  Description: 自定义的 OAOASTVisitor 和 OAOASTConsumer
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

#ifndef OAO_REWRITER_H
#define OAO_REWRITER_H

#include "Clang_LLVM_Inc.h"
#include "SrcRange.h"
#include "SeqParNodeBase.h"
#include "VarInfo.h"
#include "FuncInfo.h"
#include "OMPRegion.h"
#include "SequentialRegion.h"
#include "ASTStackNode.h"

using namespace clang;

// RecursiveASTVisitor is is the big-kahuna visitor that traverses
// everything in the AST.
class OAOASTVisitor : public RecursiveASTVisitor<OAOASTVisitor>
{
public:

    bool   BeginFlag;
    // int    indexCurrentOMP;
    // size_t LocalVarID; // 在代码转换过程中: 为定义的局部变量计数, 变量名中加入计数防止重复
    std::vector<FUNC_INFO>         vectorFunc; // 函数列表, 记录并行域中调用的函数
    std::vector<ASTStackNode>      ASTStack; // 在 AST 先深搜索中, 用来保存搜索顺序的栈
    std::vector<FuncStackNode>     FuncStack; // 这里需要定义函数栈
    std::vector<NODE_INDEX>        OMPStack; // 正在处理的 OMP 域, 考虑套嵌套, 所以建立一个栈
    // std::vector<CompoundStackNode> CompoundStack; // CompoundStmt 栈, 栈顶即为当前所在的 "{ }" 的范围, 即为变量的作用域
    std::vector<MY_SOURCE_RANGE>   CompoundStack; // CompoundStmt 栈, 栈顶即为当前所在的 "{ }" 的范围, 即为变量的作用域
    // std::vector<DeclStmt*>         pDeclStmtStack; // DeclStmt* 栈, 知道变量定义所在的 DeclStmt, 才能知道变量定义语句的结尾的 位置, 从而获得代码插入位置
    std::vector<MY_SOURCE_RANGE>   DeclStmtStack; // DeclStmt* 栈, 知道变量定义所在的 DeclStmt, 才能知道变量定义语句的结尾的 位置, 从而获得代码插入位置
    std::vector<CODE_REPLACE>      GlobalReplace; // wfr 20190729 当需要替换的代码不在任何函数中 即 FuncStack 为空时, 替换信息保存在这里
    std::vector<VARIABLE_INFO>     vectorGblVar; // wfr 20170806 保存全局变量

    ASTContext&  OAOASTContext;
    Rewriter&    Rewrite;
    //ASTDumper&   MyDumper;
    
    //OAOASTVisitor(ASTContext& C, Rewriter& R, ASTDumper& D) : OAOASTContext(C), Rewrite(R), MyDumper(D) {
    OAOASTVisitor(ASTContext& C, Rewriter& R) : OAOASTContext(C), Rewrite(R) {
        BeginFlag = false; // LocalVarID = 0; // indexCurrentOMP = -1;
    }
    
    // 以下是 override 的函数
    bool VisitReturnStmt(ReturnStmt *S); // wfr 20190430 将 return 节点的范围保存到当前节点
    bool shouldTraversePostOrder() const { return true; } // 决定是否对 AST 进行后序访问 默认是 false

    bool VisitOMPParallelForDirective(OMPParallelForDirective *D);
    bool VisitCXXNewExpr(CXXNewExpr *S); // wfr 20190729 处理 new
    // bool VisitCXXDeleteExpr(CXXDeleteExpr *S); // wfr 20190729 处理 delete
    
    bool VisitVarDecl(VarDecl *VD);
    bool VisitFunctionDecl(FunctionDecl *S);
    bool VisitCallExpr(CallExpr *S);
    bool OAOVisitDeclRefExpr(DeclRefExpr *S);
    bool OAOVisitMemberExpr(MemberExpr *S);

    bool TraverseOMPParallelForDirective(OMPParallelForDirective *S, DataRecursionQueue *Queue = nullptr);
    //bool TraverseBinaryOperator(BinaryOperator *S, DataRecursionQueue *Queue = nullptr); // BinaryOperator 相关语法树遍历 override 函数
    //bool TraverseCompoundAssignOperator(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    //bool TraverseUnaryOperator(UnaryOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseVarDecl(VarDecl *D);
    bool TraverseArraySubscriptExpr(ArraySubscriptExpr *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseImplicitCastExpr(ImplicitCastExpr *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseMemberExpr(MemberExpr *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseDeclRefExpr(DeclRefExpr *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseFunctionDecl(FunctionDecl *D);
    bool VisitParmVarDecl(ParmVarDecl *D);
    bool TraverseCallExpr(CallExpr *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseCompoundStmt( CompoundStmt *S, DataRecursionQueue *Queue = nullptr);
    bool VisitCompoundStmt(CompoundStmt *S);
    bool TraverseDeclStmt( DeclStmt *S, DataRecursionQueue *Queue = nullptr);
    bool VisitDeclStmt(DeclStmt *S);

    bool TraverseForStmt(ForStmt *S, DataRecursionQueue *Queue = nullptr);
    bool VisitForStmt(ForStmt *S);

    bool TraverseCXXDeleteExpr(CXXDeleteExpr *S, DataRecursionQueue *Queue = nullptr); // wfr 20190729 处理 delete

    

    // 以下是自定义的函数

    bool saveFuncInfo(DeclRefExpr *S, int indexOMP); // 保存 OMP 中调用的函数的信息
    bool saveLocalClassInfo(VarDecl* pClassDecl, int indexFunc, int indexOMP); // 保存 OMP 中 LocalClass 信息
    int  saveLocalMemberInfo(FieldDecl* pMemberDecl, int indexLocalClass, int indexFunc, int indexOMP); // 保存 OMP 中 LocalMember 信息
    int  saveMemberInfo(FieldDecl* pMemberDecl, int indexClass, int indexFunc); // 保存 OMP 外部 Member 信息
    int  saveMemberInfo(FieldDecl* pMemberDecl, int indexClass, FUNC_INFO& Func);
    int  saveGlobalMemberInfo(FieldDecl* pMemberDecl, int indexGblClass); // wfr 20190807 保存全局变量中的子域
    void saveVarRefInfo(std::vector<VAR_REF_LIST>& vectorVarRef, int indexVar, VAR_REF::MEMBER_TYPE in_MemberType, STATE_CONSTR in_StReq, STATE_CONSTR in_StTransFunc, MY_SOURCE_RANGE& in_SrcRange); // wfr 20190325 保存某个变量的一次引用
    bool translation(); // 在此函数中改写源文件
    NODE_INDEX SplitFuncNode(CallExpr *S);
    NODE_INDEX SplitOMPNode(Stmt* S); // wfr 20190222 在当前函数中分离出 OMP 节点, 被 TraverseOMPParallelForDirective 调用
    
    
    // wfr 20190213 判断 某个Stmt 是否在 某个OMP 中
    bool IsInOMP(Stmt* S, unsigned int indexFunc, unsigned int indexOMP);
    bool IsInOMP(Decl* S, unsigned int indexFunc, unsigned int indexOMP);
    bool IsInOMP(Expr* S, unsigned int indexFunc, unsigned int indexOMP);

    int EnqueueChildren(std::vector<NODE_INDEX>& vectorProcess, NODE_INDEX indexNode, int indexVar, FUNC_INFO& Func); // 将当前节点的在变量作用域中的子节点入队
    SEQ_PAR_NODE_BASE* getNodeBasePtr(const FUNC_INFO& Func, const NODE_INDEX& indexNode); // 根据 NODE_INDEX 的 type及index 获得相应节点的指针
    // wfr 20190720 协商 入口同步状态; 用于 确定 for/while 循环 入口处的 变量入口状态
    int  CoSyncStateBegin(STATE_CONSTR& EntryState, STATE_CONSTR& SyncStateBegin);
    int  InferParmInterfaceState(FUNC_INFO& FuncCurrent); // 先深遍历串并行图, 推断 函数参数 的 接口同步状态
    STATE_CONSTR CheckBlockGroup(std::vector<BLOCK_INFO>& vectorBlock, NODE_INDEX indexNode, int indexVar, FUNC_INFO& Func); // wfr 20190309 该函数用来初始化 阻塞组, 即找到需要与当前节点(在当前变量的情况下)一起阻塞的节点
    bool AnalyzeSEQNode(int indexSEQ, FUNC_INFO& Func); // 分析处理 SEQ 节点, 遍历变量引用类型, 分析推导出 入口出口的变量同步状态
    bool AnalyzeOMPNode(int indexOMP, FUNC_INFO& Func); // 分析处理 OMP 节点, 遍历变量引用类型, 分析推导出 入口出口的变量同步状态
    int  StateTrans(TRANS_TYPE& TransType, SYNC_STATE& destReal, SYNC_STATE src, SYNC_STATE dest); // wfr 20190312 处理两个状态之间的转换, src -> dest, 获得传输类型, 以及最终实际的状态

    int  AnalyzeAllFunctions(); // 在 AST遍历 结束后调用该函数, 分析所有函数
    bool AnalyzeOneFunction(int indexFunc); // 在 AnalyzeAllFunctions 以及 VisitFunctionDecl 函数中 调用该函数, 分析一个具体的函数

    int  getArrayLen(std::string& ArrayLen, const std::string& TypeName); // wfr 20190318 获得多维数组的元素的个数, 即数组长度
    // int  InsertVarRef(VAR_REF_LIST& VarRefList, REF_TYPE RefType, VAR_REF::MEMBER_TYPE isArrow, MY_SOURCE_RANGE RefRange); // wfr 20190320 向一个变量的引用列表中插入新引用
    void InWhichSEQOMP(NODE_INDEX& indexNode, unsigned int indexFunc, unsigned int Offset); // wfr 20190324 判断一个 Offset 在哪个 SEQ/OMP 中, 根据源代码文本顺序
    int  WhichLocalVar(Decl* ptrDecl, unsigned int indexFunc, unsigned int indexOMP); // 变量的定义是否在 vectorLocal 列表中, 以及是哪个表项,返回 vectorLocal 表项的 index, 不在就返回 -1
    int  InWhichOMP(unsigned int indexFunc, unsigned int Offset); // 该函数用来判断一个偏移是否在OMP中, 以及在哪个OMP中, 返回 vectorOMP 表项的 index 或 -1
    int  InWhichSEQ(unsigned int indexFunc, unsigned int Offset); // 该函数用来判断一个偏移是否在SEQ中, 以及在哪个SEQ中, 返回 vectorOMP 表项的 index 或 -1
    int  InWhichFunc(unsigned int Offset); // 判断一个偏移在哪个函数中
    int  WhichFuncDecl(FunctionDecl* pFuncDecl); // 该函数用来判断一个函数是否在 vectorFunc 中, 以及是哪个 FUNC_INFO , 返回 vectorFunc 表项的 index 或 -1
    int  WhichFuncDefine(FunctionDecl* pFuncDecl); // 判断一个 函数的定义 是否在 vectorFunc 中, 在则返回 index, 不在则返回 -1
    int  WhichVar(Decl* ptrDecl, unsigned int indexFunc); // 变量的定义是否在变量列表中, 以及是哪个表项,返回 vectorVar 表项的 index
    int  WhichGblVar(Decl* ptrDecl); // wfr 20190809 返回全局变量在 vactorGblVar 中的 index, 没找到就返回 -1
    int  MemberIsWhichLocal(Decl* ptrClass, std::string Name, unsigned int indexFunc, unsigned int indexOMP); // 检查 member 是否在变量列表中, 以及是哪个表项,返回 vectorVar 表项的 index
    int  MemberIsWhichVar(Decl* ptrClass, std::string Name, unsigned int indexFunc); // 检查 member 是否在变量列表中, 以及是哪个表项,返回 vectorVar 表项的 index
    int  MemberIsWhichVar(Decl *ptrClass, std::string Name, FUNC_INFO& Func);
    int  MemberIsWhichGlobal(Decl *ptrClass, std::string Name); // wfr 20190807 变量是哪个全局 member
    void InitVar(VARIABLE_INFO& Var, VarDecl* ptr, int indexFunc, NODE_INDEX indexDefNode); // wfr 20190330 初始化 VARIABLE_INFO
    void ScopeBeginEndNode(NODE_INDEX& indexOut, int indexFunc, MY_SOURCE_RANGE Scope, SCOPE_BEGIN_END flag); // wfr 20190409 获得一个 Scope 的 开始节点/结束节点 的 NODE_INDEX
    NODE_INDEX FindDefNode(const FUNC_INFO& Func, const VARIABLE_INFO& Var); // wfr 20190415 遍历 vectorSEQ, 找到某个 OMP 外部变量 定义在哪个 SEQ 中
    // wfr 20190725 推断 while/for循环 整体的 入口状态需求 和 等效的状态转换函数
    int  InferLoopInterfaceState(STATE_CONSTR& LoopStReq, STATE_CONSTR& LoopStTrans, 
        bool& LoopNeedAnalysis, NODE_INDEX indexLoopEntry, int indexVar, FUNC_INFO& Func, 
        bool& outStTransHostWrite, bool& outStTransDeviceWrite, bool& outStReqHostNew, bool& outStReqDeviceNew);
    FunctionDecl* getCalleeFuncPtr(CallExpr* pCallExpr); // wfr 20190613 获得被调函数定义地址
    // wfr 20190716 对于在 OMP中 调用函数 的情况, 对 StReq 和 StTransFunc 进行转换
    void ModifyFuncCallInOMP(STATE_CONSTR& StReq, STATE_CONSTR& StTransFunc);
    // wfr 20190716 检查 StReq 是否被初始化了
    void CheckStReq(STATE_CONSTR& in);
    // wfr 20190716 检查 StTrans 是否被初始化了
    void CheckStTrans(STATE_CONSTR& in);
    // wfr 20190716 执行同步状态装换
    SYNC_BITS ExeStTrans(const STATE_CONSTR& StTransFunc, const SYNC_BITS& SyncBits);
    // wfr 20190716 执行同步状态装换
    STATE_CONSTR ExeStTrans(const STATE_CONSTR& StTransFunc, const STATE_CONSTR& StReq);
    // wfr 20190719 处理节点入口处的变量状态转换相关问题: 
    // 1. 从 src(即前驱节点的出口状态约束) 到 dest(即当前节点的入口状态约束) 是否需要插入 OAODataTrans函数
    // 2. destReal(即实际的入口状态约束) 是什么
    int InferEnteryTrans(STATE_CONSTR &StTrans, STATE_CONSTR &destReal, STATE_CONSTR src, STATE_CONSTR dest);
    // wfr 20190719 处理节点入口处的变量状态转换相关问题:
    // 1. ExitStConstr(即出口状态约束) 是什么
    // 2. 出口处是否插入 OMPWrite/SEQWrite函数, 由 ExitStTrans 标识
    int InferExitTrans(STATE_CONSTR &ExitStTrans, STATE_CONSTR &ExitStConstr, STATE_CONSTR EntryStConstr, STATE_CONSTR StTrans);
    // wfr 20190724 对 函数的 UsedInOMP 进行传播, 标记出所有运行在 device 上的函数, 这些函数要保守分析
    int SpreadFuncUsedInOMP(FUNC_INFO& Func);
    // wfr 20200114 对 变量的 UsedInOMP 进行传播, 父函数中变量在 device 上被使用, 则在被调用函数中 UsedInOMP=true, 表示该变量也需要分析
    int SpreadVarUsedInOMP(FUNC_INFO& Func);
    // wfr 20190725 找到 循环体的第一个节点 和 循环之后的第一个节点
    int MarkLoopBranch(FUNC_INFO& Func);
    // wfr 20190726 求两个状态约束/需求的交集, 不存在 就返回 StReqUninit
    STATE_CONSTR ReqIntersection(STATE_CONSTR& StReq0, STATE_CONSTR& StReq1);
    // wfr 20190817 推导从同步状态约束 src 到同步状态约束 dest 的转换函数
    STATE_CONSTR InferStTransFunc(STATE_CONSTR& src, STATE_CONSTR& dest);
    // wfr 20190726 检查 同步状态 是否 满足约束
    bool CheckConstrSatisfaction(STATE_CONSTR& state, STATE_CONSTR& constraint);
    // wfr 20190806 通过变量类型字符串, 分析获得变量类型
    VAR_TYPE getVarType(std::string& TypeName);
    // wfr 20190809 获得 某函数member/全局member(不包括并行域中定义的member) 的 index 如果变量还不存在就存入
    int  getMemberIndex(Decl* ptrClass, FieldDecl* pMemberDecl, unsigned int indexFunc);
    // wfr 20190808 获得 某函数中变量/全局变量(不包括并行域中定义的变量) 的 index 如果变量还不存在就存入
    int  getVarIndex(Decl* pVarDecl, unsigned int indexFunc);
    
    // ghn 20200301 ForStmt 深度遍历
    void DeepTraversal(Stmt* pParent, OMP_REGION& OMP);
    
    // ghn 20200301 ForStmt第一个子节点的遍历
    void ForStmtFirstTraversal(Stmt* pBinary, std::vector<std::string>& declRefe);


    //friend class MY_SOURCE_RANGE;
    //friend class VARIABLE_INFO;
    //friend class FUNC_INFO;
    //friend class OMP_REGION;
    //friend class RecursiveASTVisitor<OAOASTVisitor>;

    bool TraverseStmt(Stmt *S, DataRecursionQueue *Queue = nullptr);

    bool TraverseUnaryAddrOf(UnaryOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseUnaryPostInc(UnaryOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseUnaryPreInc(UnaryOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseUnaryPostDec(UnaryOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseUnaryPreDec(UnaryOperator *S, DataRecursionQueue *Queue = nullptr);

    bool TraverseBinAssign(BinaryOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinMulAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinDivAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinRemAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinAddAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinSubAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinShlAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinShrAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinAndAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinXorAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);
    bool TraverseBinOrAssign(CompoundAssignOperator *S, DataRecursionQueue *Queue = nullptr);

// private:
//     template<typename T, typename U>
//     struct has_same_member_pointer_type : std::false_type {};
//     template<typename T, typename U, typename R, typename... P>
//     struct has_same_member_pointer_type<R (T::*)(P...), R (U::*)(P...)>
//         : std::true_type {};
/*private:
    template<typename T, typename U>
    struct has_same_member_pointer_type : std::false_type {};
    template<typename T, typename U, typename R, typename... P>
    struct has_same_member_pointer_type<R (T::*)(P...), R (U::*)(P...)>
        : std::true_type {};
    bool TraverseFunctionHelper(FunctionDecl *D);
    bool TraverseTemplateArgumentLocsHelper(const TemplateArgumentLoc *TAL, unsigned Count);
    bool TraverseVarHelper(VarDecl *D);
    bool TraverseDeclContextHelper(DeclContext *DC);
    bool TraverseOMPExecutableDirective(OMPExecutableDirective *S);
    bool TraverseTemplateParameterListHelper(TemplateParameterList *TPL);
    template <typename T> bool TraverseDeclTemplateParameterLists(T *D);
    bool TraverseDeclaratorHelper(DeclaratorDecl *D);
    bool TraverseOMPClause(OMPClause *C);
    bool VisitOMPIfClause(OMPIfClause *C);*/
};


class OAOASTConsumer : public ASTConsumer
{
public:
    OAOASTVisitor ASTVisitor;

    OAOASTConsumer(ASTContext& C, Rewriter &R) : ASTVisitor(C, R) { }
    virtual bool HandleTopLevelDecl(DeclGroupRef D);
};


#endif
